from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
from langfuse.openai import OpenAI

lf = Langfuse()
assert lf.auth_check(), "❌ Credenciales Langfuse inválidas — revisa LANGFUSE_PUBLIC_KEY y LANGFUSE_SECRET_KEY"
print("✅ Auth OK")

if __name__ == "__main__":
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Que es una Acometida?"}],
        name="test-trace",
        metadata={"component": "test_graph"},
        tags=["test", "langfuse"],
    )

    print("Respuesta:", response.choices[0].message.content)

    lf.flush()
    print("✅ Trazas enviadas — verifica en Langfuse UI (https://cloud.langfuse.com)")
