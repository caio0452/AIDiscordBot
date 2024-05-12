from dataclasses import dataclass
import datetime
import discord

@dataclass
class MemorizedMessage:
    text: str
    nick: str
    sent: datetime.datetime
    is_bot: bool
    message_id: int

    def __str__(self):
        formatted_time = datetime.datetime.strftime(self.sent, "%Y-%m-%d %H:%M:%S")
        return f"[{formatted_time}] {self.nick}: {self.text}"

    @staticmethod
    async def of_discord_message(message: discord.Message, message_sanitizer = None) -> "MemorizedMessage":
        if message_sanitizer is not None:
            text = await message_sanitizer(message)
        else:
            text = message.content

        return MemorizedMessage(
            text=f"[{message.created_at.strftime('%d/%m %H:%M:%S')} by {message.author.display_name}] {text}",
            nick=message.author.display_name,
            sent=message.created_at,
            is_bot=message.author.bot,
            message_id=message.id
        )