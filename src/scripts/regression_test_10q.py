import requests
import json
import time

BASE_URL = "http://localhost:8000"
KB_ID = "test-kb"

QUESTIONS = [
    "What is the role of a software tester?",
    "How do unit tests differ from integration tests?",
    "What are best practices for debugging a memory leak?",
    "Who won the Oscar for Best Picture in 2024?",
    "What is the current price of Bitcoin?",
    "What are the latest updates on the James Webb Space Telescope?",
    "How does the knowledge base suggest handling flaky tests?",
    "What is the weather currently in Tokyo?",
    "Who is the current CEO of Microsoft?",
    "What are the primary advantages of safety-critical systems mentioned in the docs?"
]

def query_kb(text, web="off"):
    url = f"{BASE_URL}/kb/{KB_ID}/query"
    payload = {
        "query": text,
        "web": web
    }
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=60)
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            return {"error": response.status_code, "text": response.text}
        
        result = response.json()
        citations = result.get("citations", [])
        web_citations = [c for c in citations if "http" in c.get("source", "")]
        
        return {
            "answer_preview": result['answer'][:150].replace("\n", " ") + "...",
            "citations_count": len(citations),
            "web_citations_count": len(web_citations),
            "time": round(elapsed, 2)
        }
    except Exception as e:
        return {"error": str(e)}

def run_regression():
    print(f"{'Question':<50} | {'Web':<5} | {'Cits':<5} | {'WebCits':<7} | {'Time':<5}")
    print("-" * 85)
    
    results = []
    for q in QUESTIONS:
        # Test with web=off
        res_off = query_kb(q, web="off")
        print(f"{q[:48]:<50} | {'OFF':<5} | {res_off.get('citations_count', 'ERR'):<5} | {res_off.get('web_citations_count', 'ERR'):<7} | {res_off.get('time', 'ERR'):<5}")
        
        # Test with web=on
        res_on = query_kb(q, web="on")
        print(f"{q[:48]:<50} | {'ON':<5} | {res_on.get('citations_count', 'ERR'):<5} | {res_on.get('web_citations_count', 'ERR'):<7} | {res_on.get('time', 'ERR'):<5}")
        print("-" * 85)
        
        results.append({"question": q, "off": res_off, "on": res_on})

    # Save detailed results to a file
    with open("regression_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to regression_results.json")

if __name__ == "__main__":
    run_regression()
