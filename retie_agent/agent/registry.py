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



# Agente único: todo el corpus (NTC 2050 V2 + erratas + RETIE Libros 1-4) vive
# en la colección "normativas" y la respuesta es híbrida (Chroma + NotebookLM).
# La vigencia de cada documento queda en la metadata `vigente` de cada chunk.
AGENTS: Dict[str, AgentConfig] = {
    "normativas": AgentConfig(
        key="normativas",
        collection="normativas",
        chat_model="gpt-4o-mini",
        top_k=4,
        system_prompt="Eres un asistente experto en normativa eléctrica colombiana (RETIE y NTC 2050). Sé preciso y cita [archivo, página].",
    ),
}
