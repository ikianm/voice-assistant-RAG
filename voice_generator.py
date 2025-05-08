from elevenlabs.client import ElevenLabs

import tempfile

class VoiceGenerator:
    def __init__(self, api_key):
        self.client = ElevenLabs(api_key=api_key)
        # Default available voices
        self.available_voices = [
            "Rachel",
            "Domi",
            "Bella",
            "Antoni",
            "Elli",
            "Josh",
            "Arnold",
            "Adam",
            "Sam",
        ]
        self.default_voice = "Rachel"  
        
    def generate_voice_response(self, text: str, voice_name: str = None) -> str:
        """Generate voice response, returns filename of the voice (.mp3)"""
        try:
            selected_voice = voice_name or self.default_voice
            audio_generator = self.client.generate(
                text=text, voice=selected_voice, model='eleven_multilingual_v2'
            )
            audio_bytes = b''.join(audio_generator)
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                return temp_audio.name
            
        except Exception as e:
            print(f"Error generating voice response: {e}")
            return None