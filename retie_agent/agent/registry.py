#Creacion de los agentes necesarios para dividir las tareas dependiendo de la preegunta que el usuario realice
# esta va a ser tomada por el orchetrator y se le envía al agente, dependiendo de su rol
from dataclasses import dataclass
from typing import Optional, Callable, Dict
from retie_agent.config import settings

@dataclass
class AgentConfig:
    key: str                       
    collection: str                
    chat_model: Optional[str] = None   
    embedding_model: Optional[str] = None  # 
    top_k: Optional[int] = None
    system_prompt: Optional[str] = None   
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    def resolved_chat_model(self) -> str:
        return self.chat_model or settings.CHAT_MODEL

    def resolved_top_k(self) -> int:
        return self.top_k or settings.TOP_K

    def resolved_max_tokens(self) -> int:
        return self.max_tokens or settings.MAX_TOKENS



AGENTS: Dict[str, AgentConfig] = {
    # Agente basado en PDFPlumber 
    "plumber": AgentConfig(
        key="plumber",
        collection="retie_docs",
        chat_model="gpt-4o-mini",    
        top_k=4,
        system_prompt="Eres un asistente experto en RETIE. Sé preciso y cita [archivo, página]."
    ),
    # Agente PyMuPDF
    "pymupdf": AgentConfig(
        key="pymupdf",
        collection="retie_pymupdf",
        chat_model="gpt-4o-mini",          
        top_k=4,
        system_prompt="Eres un asistente técnico. Responde completo y con citas exactas [archivo, página]."
    ),

}
