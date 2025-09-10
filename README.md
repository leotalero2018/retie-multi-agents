
# python-telegram-bot-RETIE
Este proyecto implementa un sistema de **agentes inteligentes** (multiagente) que responden preguntas técnicas sobre el RETIE (Reglamento Técnico de Instalaciones Eléctricas).

# retie-agent

Bot de Telegram con RAG (Chroma + OpenAI) para responder preguntas sobre documentos RETIE.

![RETIE Flow](retie-agent\images\diagrama.png)





## 1 Preparar entorno

```bash

CLONACION DEL REPOSITORIO

git clone https://github.com/leotalero2018/retie-multi-agents.git
cd retie-multi-agents

CREACION DEL ENTORNO VIRTUAL

PARA LINUX/MAC

python3 -m venv venv
source venv/bin/activate

PARA WINDOWS

python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate

INSTALACION DE DEPENDENCIAS (LIBRERIAS)

pip install -r requirements.txt

EJECUCION DEL ENTORNO

PARA LINUX/MAC

export PYTHONPATH=$(pwd)

PARA WINDOWS

$env:PYTHONPATH = "$(Get-Location)"

PRUEBA LOCAL (LA RESPUESTA ES MUY BASICA POR EL MOMENTO)

python -c "from app.agent.orchestrator import answer_question; print(answer_question('¿Qué exige el RETIE sobre puesta a tierra en subestaciones?'))"

PRUEBA CON TELEGRAM

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$env:PYTHONPATH = "$(Get-Location)"
python -m app.bot.run_polling

Estructura del proyecto

retie-agent/
├─ app/
│ ├─ agent/ # Orquestador RAG/LLM y herramientas
│ │ ├─ init.py
│ │ ├─ orchestrator.py
│ │ ├─ prompt.py
│ │ ├─ tools.py
│ │ └─ graph.py
│ │
│ ├─ api/ # API con FastAPI (para integración / webhooks)
│ │ ├─ init.py
│ │ └─ main.py
│ │
│ ├─ bot/ # Bot de Telegram (aiogram)
│ │ ├─ init.py
│ │ └─ run_polling.py
│ │
│ ├─ ingestion/ # Extracción de texto, chunking, embeddings, indexado
│ │ ├─ init.py
│ │ ├─ chunker.py
│ │ ├─ embedder.py
│ │ ├─ extractors.py
│ │ └─ indexer.py
│ │
│ ├─ llm/ # Capa de conexión con modelos LLM
│ │ ├─ init.py
│ │ └─ provider.py
│ │
│ ├─ observability/ # Monitoreo y observabilidad
│ │ └─ obs.py
│ │
│ ├─ retriever/ # Conexión y búsquedas en Chroma
│ │ ├─ init.py
│ │ ├─ chroma_client.py
│ │ └─ retrieve.py
│ │
│ ├─ utils/ # Utilidades comunes
│ │ ├─ init.py
│ │ ├─ text.py
│ │ └─ config.py
│ │
│ └─ init.py
│
├─ data/ # Documentos fuente y DB persistente de Chroma
│ ├─ chroma_db/ # Persistencia de embeddings
│ ├─ 73f9a4a0...db/ # Carpeta autogenerada por Chroma
│ └─ chroma.sqlite3 # Base de datos local
│
├─ docs/ # Documentos normativos RETIE/NTC
│ ├─ 2_Libro_1__Disposiciones_Generales.pdf
│ ├─ NTC_2050_V2_Codigo_Electrico_Colombiano.pdf
│ ├─ NTC_2050-Fe-de-erratas.pdf
│ └─ Resolución_40117_de_2024_RETIE.pdf
│
├─ tests/ # Pruebas unitarias
│ ├─ init.py
│ └─ test_chunker.py
│
├─ .env.example # Ejemplo de variables de entorno
├─ .gitignore
├─ requirements.txt # Dependencias del proyecto
├─ Dockerfile # Imagen de contenedor
├─ docker-compose.yml # Orquestación con contenedores
├─ index_docs.py # Script CLI para indexar documentos
└─ README.md
