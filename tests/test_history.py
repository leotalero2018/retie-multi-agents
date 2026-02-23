# tests/test_history.py
import os
import sys
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agent.graph import run_graph
import uuid

def test_conversation_history():
    session_id = f"test-session-{uuid.uuid4()}"
    user_id = "test-user"
    
    print(f"--- Starting test with session {session_id} ---")
    
    # First question
    q1 = "Hola, mi nombre es Leonardo."
    print(f"User: {q1}")
    r1 = run_graph(q1, user_id=user_id, session=session_id)
    print(f"Assistant: {r1}")
    
    # Second question (referring to the first)
    q2 = "¿Sabes cuál es mi nombre?"
    print(f"User: {q2}")
    r2 = run_graph(q2, user_id=user_id, session=session_id)
    print(f"Assistant: {r2}")
    
    # Verification
    if "Leonardo" in str(r2):
        print("\n✅ Success: Assistant remembers the user's name!")
    else:
        print("\n❌ Failure: Assistant does NOT remember the user's name.")
        # print("Final answer was:", r2)

if __name__ == "__main__":
    test_conversation_history()
