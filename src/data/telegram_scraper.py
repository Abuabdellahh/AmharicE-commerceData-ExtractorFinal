import asyncio
import json
import pandas as pd
from telethon import TelegramClient
from datetime import datetime, timedelta
import re
import logging
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class TelegramDataIngestion:
    def __init__(self, api_id: str, api_hash: str, phone: str):
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.client = TelegramClient('session', api_id, api_hash)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def connect(self):
        """Connect to Telegram"""
        await self.client.start(phone=self.phone)
        self.logger.info("Connected to Telegram")
        
    async def get_channel_messages(self, channel_username: str, limit: int = 1000) -> List[Dict]:
        """Fetch messages from a Telegram channel"""
        messages = []
        
        try:
            entity = await self.client.get_entity(channel_username)
            
            async for message in self.client.iter_messages(entity, limit=limit):
                if message.text:
                    message_data = {
                        'id': message.id,
                        'text': message.text,
                        'date': message.date.isoformat(),
                        'views': message.views or 0,
                        'forwards': message.forwards or 0,
                        'channel': channel_username,
                        'sender_id': message.sender_id,
                        'media_type': 'text'
                    }
                    
                    # Handle media messages
                    if message.media:
                        message_data['media_type'] = str(type(message.media).__name__)
                        if hasattr(message.media, 'photo'):
                            message_data['has_photo'] = True
                    
                    messages.append(message_data)
                    
        except Exception as e:
            self.logger.error(f"Error fetching messages from {channel_username}: {e}")
            
        return messages
    
    async def scrape_multiple_channels(self, channels: List[str], limit_per_channel: int = 1000) -> pd.DataFrame:
        """Scrape messages from multiple channels"""
        all_messages = []
        
        for channel in channels:
            self.logger.info(f"Scraping channel: {channel}")
            messages = await self.get_channel_messages(channel, limit_per_channel)
            all_messages.extend(messages)
            
            # Add delay to avoid rate limiting
            await asyncio.sleep(2)
            
        df = pd.DataFrame(all_messages)
        return df
    
    async def close(self):
        """Close the Telegram client"""
        await self.client.disconnect()

# Ethiopian e-commerce channels (example list)
ETHIOPIAN_CHANNELS = [
    '@shageronlinestore',
    '@ethio_market_place',
    '@addis_shopping',
    '@bole_electronics',
    '@merkato_online'
]

async def main():
    # Initialize scraper
    scraper = TelegramDataIngestion(
        api_id=os.getenv('TELEGRAM_API_ID'),
        api_hash=os.getenv('TELEGRAM_API_HASH'),
        phone=os.getenv('TELEGRAM_PHONE')
    )
    
    await scraper.connect()
    
    # Scrape data
    df = await scraper.scrape_multiple_channels(ETHIOPIAN_CHANNELS)
    
    # Save raw data
    df.to_csv('data/raw/telegram_messages.csv', index=False)
    df.to_json('data/raw/telegram_messages.json', orient='records', indent=2)
    
    print(f"Scraped {len(df)} messages from {len(ETHIOPIAN_CHANNELS)} channels")
    
    await scraper.close()

if __name__ == "__main__":
    asyncio.run(main())
