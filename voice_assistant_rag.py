from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

import whisper

import sounddevice as sd
import soundfile as sf

import tempfile
import os

from voice_generator import VoiceGenerator

class VoiceAssistantRAG:
    def __init__(self, elevenlabs_api_key: str):
        self.llm = ChatOllama(model='llama3.2')
        self.embeddings = OllamaEmbeddings(model='nomic-embed-text')
        self.whisper = whisper.load_model('base')
        self.vector_store: Chroma = None
        self.qa_chain = None
        self.sample_rate = 44100
        self.voice_generator = VoiceGenerator(api_key=elevenlabs_api_key)
    
    def setup_qa_chain(self, vector_store: Chroma) -> None:
        """Initialize the vector store & QA chain"""
        self.vector_store = vector_store
        
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            memory=memory,
            retriever=self.vector_store.as_retriever(),
            llm=self.llm
        )
        
    def record_audio(self, duration=5):
        """Record audio from microphone"""
        recording_byte_array = sd.rec(
            frames=int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1
        )
        sd.wait()
        return recording_byte_array
    
    def transcribe_audio(self, recording_byte_array):
        """Transcribe audio using Whisper"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
            sf.write(file=temp_audio_file, data=recording_byte_array, samplerate=self.sample_rate)
            result = self.whisper.transcribe(temp_audio_file.name)
            os.unlink(temp_audio_file.name)
        return result['text']
        
    def generate_response(self, query: str):
        """Generate response using RAG system"""
        if self.qa_chain is None:
            return "Error: Vector store not initialized"
        
        response = self.qa_chain.invoke({'question': query})
        return response['answer']
    
    def text_to_speech(self, text: str, voice_name: str = None) -> str:
        """Convert text to speech"""
        return self.voice_generator.generate_voice_response(text, voice_name)  
        
    