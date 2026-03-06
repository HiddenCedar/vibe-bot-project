#!/usr/bin/env python3
"""
Vibe Bot - Telegram AI assistant using free local models
No API keys required - uses Ollama (recommended) or Hugging Face Transformers
"""
import os
import sys
import asyncio
import logging
from collections import defaultdict
from typing import Optional, List, Tuple
from dataclasses import dataclass

# Third-party imports
try:
    from telegram import Update
    from telegram.ext import (
        Application, CommandHandler, MessageHandler, 
        filters, ContextTypes, ApplicationBuilder
    )
    import httpx
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install python-telegram-bot httpx python-dotenv")
    sys.exit(1)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
# Load from .env file if present
if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    logger.error("BOT_TOKEN not set in environment or .env file")
    sys.exit(1)

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
AI_BACKEND = os.getenv("AI_BACKEND", "auto").lower()  # auto, ollama, or transformers

# ==================== AI CLIENT ABSTRACTION ====================
@dataclass
class AIResponse:
    content: str
    backend: str
    response_time: float

class AIClient:
    """Abstract AI client with multiple backend support."""
    
    def __init__(self):
        self.backend = None
        self.model_name = None
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the appropriate backend based on config/availability."""
        if AI_BACKEND in ("ollama", "auto"):
            if self._try_init_ollama():
                return
        
        if AI_BACKEND in ("transformers", "auto"):
            if self._try_init_transformers():
                return
        
        raise RuntimeError(
            "No AI backend available. Install Ollama (https://ollama.ai) "
            "or ensure transformers dependencies are installed."
        )

    async def warmup(self):
        """Warm up the model with a simple inference to reduce first-call latency."""
        logger.info("Warming up AI model...")
        try:
            await self.generate("Hello", [])
            logger.info("✅ Model warmed up and ready")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def _try_init_ollama(self) -> bool:
        """Try to initialize Ollama client."""
        try:
            import ollama
            # Test connection
            try:
                response = ollama.list()
            except Exception as conn_e:
                logger.warning(f"Ollama connection failed: {conn_e}")
                return False
            if response.get('models'):
                self.backend = "ollama"
                self.model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
                logger.info(f"✅ Using Ollama backend with model: {self.model_name}")
                return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
        return False
    
    def _try_init_transformers(self) -> bool:
        """Try to initialize Transformers client."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_name = os.getenv("HF_MODEL", "microsoft/DialoGPT-medium")
            device = 0 if torch.cuda.is_available() else -1
            
            logger.info(f"Loading HuggingFace model: {model_name}...")
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backend = "transformers"
            self.model_name = model_name
            logger.info(f"✅ Using Transformers backend with model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Transformers initialization failed: {e}")
        return False
    
    async def generate(self, prompt: str, history: List[Tuple[str, str]]) -> AIResponse:
        """Generate a response from the AI."""
        if self.backend == "ollama":
            return await self._generate_ollama(prompt, history)
        elif self.backend == "transformers":
            return await self._generate_transformers(prompt, history)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    async def _generate_ollama(self, prompt: str, history: List[Tuple[str, str]]) -> AIResponse:
        """Generate using Ollama with streaming disabled for complete responses."""
        import ollama
        import time
        
        start = time.time()
        
        # Build conversation history for Ollama (keep minimal)
        messages = []
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Get configured max tokens and add small buffer to avoid cutoff
            max_tokens = int(os.getenv("MAX_TOKENS", "150"))
            # Add 20% buffer to ensure completion
            effective_tokens = int(max_tokens * 1.2)
            
            # IMPORTANT: stream=False to get complete response in one go
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=False,  # Get full response, not chunks
                options={
                    "temperature": float(os.getenv("TEMPERATURE", "0.25")),
                    "num_predict": effective_tokens,
                    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "1024")),
                    "top_p": float(os.getenv("TOP_P", "0.9")),
                    "top_k": int(os.getenv("TOP_K", "25")),
                    "num_thread": int(os.getenv("OLLAMA_NUM_THREADS", "8"))
                }
            )
            content = response["message"]["content"].strip()
            elapsed = time.time() - start
            return AIResponse(content=content, backend="ollama", response_time=elapsed)
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return AIResponse(
                content="Sorry, I encountered an error with the AI service.",
                backend="ollama",
                response_time=0
            )
            content = response["message"]["content"].strip()
            elapsed = time.time() - start
            return AIResponse(content=content, backend="ollama", response_time=elapsed)
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return AIResponse(
                content="Sorry, I encountered an error with the AI service.",
                backend="ollama",
                response_time=0
            )
    
    async def _generate_transformers(self, prompt: str, history: List[Tuple[str, str]]) -> AIResponse:
        """Generate using Transformers pipeline."""
        import time
        import torch
        
        start = time.time()
        
        # Build context
        context = ""
        for user_msg, bot_msg in history:
            context += f"User: {user_msg}\nBot: {bot_msg}\n"
        context += f"User: {prompt}\nBot:"
        
        try:
            # Generate with optimized parameters for speed
            max_new_tokens = int(os.getenv("MAX_TOKENS", "150"))
            with torch.no_grad():
                outputs = self.generator(
                    context,
                    max_new_tokens=max_new_tokens,
                    temperature=float(os.getenv("TEMPERATURE", "0.25")),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=float(os.getenv("TOP_P", "0.9")),
                    top_k=int(os.getenv("TOP_K", "25"))
                )
            
            generated_text = outputs[0]["generated_text"]
            # Extract only the new part after the prompt
            response_text = generated_text[len(context):].strip()
            # Clean up any trailing user prompts or incomplete sentences
            for stop_seq in ["\nUser:", "\nBot:", "</s>", "[INST]"]:
                if stop_seq in response_text:
                    response_text = response_text.split(stop_seq)[0].strip()
            
            elapsed = time.time() - start
            return AIResponse(
                content=response_text or "I'm not sure how to respond.",
                backend="transformers",
                response_time=elapsed
            )
        except Exception as e:
            logger.error(f"Transformers generation error: {e}")
            return AIResponse(
                content="Sorry, I encountered an error generating a response.",
                backend="transformers",
                response_time=0
            )

# ==================== BOT HANDLERS ====================
conversation_history = defaultdict(list)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await update.message.reply_text(
        "🤖 **Vibe Bot**\n\n"
        "I'm your AI-powered Telegram assistant!\n"
        "I work completely locally - no API keys needed.\n\n"
        "**Commands:**\n"
        "/start - Show this message\n"
        "/help - Show help & features\n"
        "/clear - Reset conversation history\n"
        "/status - Check AI backend status\n\n"
        "Just send me a message to chat!",
        parse_mode="Markdown"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_text = (
        "🤖 **Vibe Bot Help**\n\n"
        "**How to use:**\n"
        "• Send any text message to have a conversation\n"
        "• I remember up to {} messages in our chat\n"
        "• Use /clear to start fresh\n\n"
        "**What I can do:**\n"
        "• Answer questions on any topic\n"
        "• Help with coding & debugging\n"
        "• Assist with writing & analysis\n"
        "• General conversation\n\n"
        "**AI Backend:**\n"
        "• Uses free local models (Ollama or Hugging Face)\n"
        "• No API costs, no rate limits\n"
        "• Full privacy - everything stays local\n\n"
        "**Note:** Responses may take 1-10 seconds depending on model and hardware."
    ).format(MAX_HISTORY)
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear user's conversation history."""
    user_id = update.effective_user.id
    if user_id in conversation_history:
        conversation_history[user_id].clear()
        await update.message.reply_text("✅ Conversation history cleared!")
    else:
        await update.message.reply_text("No history to clear.")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show bot and AI backend status."""
    try:
        ai_client = context.bot_data.get("ai_client")
        if not ai_client:
            await update.message.reply_text("❌ AI client not initialized")
            return
        
        status_msg = (
            "🤖 **Vibe Bot Status**\n\n"
            f"**AI Backend:** {ai_client.backend}\n"
            f"**Model:** {ai_client.model_name}\n"
            f"**Max History:** {MAX_HISTORY} messages\n"
            f"**Active Users:** {len(conversation_history)}\n\n"
            "✅ All systems operational"
        )
        await update.message.reply_text(status_msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Status error: {e}")
        await update.message.reply_text("❌ Could not retrieve status.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages."""
    user_id = update.effective_user.id
    user_message = update.message.text
    
    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )
    
    try:
        ai_client: AIClient = context.bot_data["ai_client"]
        history = conversation_history[user_id]
        
        # Generate response
        response = await ai_client.generate(user_message, history)
        
        # Save to history
        history.append((user_message, response.content))
        if len(history) > MAX_HISTORY:
            history.pop(0)
        
        # Send response
        await update.message.reply_text(response.content)
        
        # Log timing
        logger.debug(f"Response generated in {response.response_time:.2f}s using {response.backend}")
        
    except Exception as e:
        logger.error(f"Error handling message from {user_id}: {e}")
        await update.message.reply_text(
            "⚠️ Sorry, I encountered an error. Please try again."
        )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Global error handler."""
    logger.error(f"Update {update} caused error: {context.error}", exc_info=context.error)
    if update and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ An internal error occurred. The issue has been logged."
            )
        except:
            pass

# ==================== MAIN ====================
def main():
    """Start the bot."""
    # Validate token
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        logger.error("Please set BOT_TOKEN in .env file")
        print("❌ BOT_TOKEN not configured. Edit .env file in current directory.")
        sys.exit(1)
    
    # Initialize AI client
    try:
        ai_client = AIClient()
        logger.info(f"AI backend initialized: {ai_client.backend} - {ai_client.model_name}")
        
        # Warm up the model (first inference is slow)
        try:
            import asyncio
            asyncio.run(ai_client.warmup())
        except Exception as e:
            logger.warning(f"Warmup skipped: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize AI client: {e}")
        print("❌ Failed to initialize AI. Make sure Ollama is installed and running, or install transformers dependencies.")
        sys.exit(1)
    
    # Build application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Store AI client in bot_data for access in handlers
    application.bot_data["ai_client"] = ai_client
    
    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("status", status_command))
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Error handler
    application.add_error_handler(error_handler)
    
    # Start bot
    logger.info("🚀 Starting Vibe Bot...")
    print("✅ Bot is running. Press Ctrl+C to stop.")
    
    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        print("\n🛑 Bot stopped.")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        print(f"❌ Bot crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
