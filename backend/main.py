from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
import re
import logging

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class EnhancedEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        
    def __call__(self, texts):
        return self.model.encode(texts).tolist()

client = Groq(api_key="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")
embedder = EnhancedEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./chroma_db_final")

class Query(BaseModel):
    query: str

class Assessment(BaseModel):
    name: str
    url: str
    remote_support: str
    adaptive_support: str 
    duration: int
    test_type: List[str]

def initialize_chroma():
    collection = chroma_client.get_or_create_collection(
        name="assessments_final_final",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"}
    )
    
    if not collection.count():
        with open("C:\\Users\\Anmol Upadhyay\\new_shl_project\\data\\products.json", encoding="utf-8") as f:
            products = json.load(f)
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, product in enumerate(products):
            doc_text = (
                f"{product['description']} "
                f"{product['name']} "
                f"{' '.join(product['test_type'])} "
                f"{' '.join(product.get('job_levels', []))} "
                f"{' '.join(product.get('Languages', []))}"
            )
            
            documents.append(doc_text)
            metadatas.append({
                "name": product["name"],
                "url": product["url"],
                "duration": product["duration"],
                "remote": product["remote_support"],
                "adaptive": product["adaptive_support"],
                "test_types": "|".join(product["test_type"]),
                "job_levels": "|".join(product.get("job_levels", [])),
                "description": product["description"]
            })
            ids.append(str(idx))
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Initialized ChromaDB with {len(products)} assessments")
    
    return collection

collection = initialize_chroma()

def enhance_query(query: str) -> dict:
    prompt = f"""Analyze this hiring query and extract:
    1. Primary job role
    2. Key technical skills (comma-separated)
    3. Test type required from: "Ability & Aptitude", "Biodata & Situational Judgement", "Competencies", "Development & 360", 
       "Assessment Exercises", "Knowledge & Skills", "Personality & Behavior", "Simulations" (comma-separated)
    4. Required experience level from: Director, Entry-Level, Executive, Front Line Manager, General Population, 
       Graduate, Manager, Mid-Professional, Professional Individual Contributor, Supervisor (comma-separated)
    5. Maximum duration in minutes
    6. Special requirements 
    
    Query: {query}
    
    Respond in JSON format. Example:
    {{
        "role": "Java Developer",
        "skills": "Java, Collaboration",
        "test_type": "Ability & Aptitude, Competencies",
        "experience": "Mid-Professional",
        "max_duration": 40,
        "requirements": "team collaboration"
    }}"""
    
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        logger.error(f"Query analysis failed: {str(e)}")
        return {
            "role": "",
            "skills": "",
            "test_type": "",
            "experience": "",
            "max_duration": 60,
            "requirements": ""
        }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/recommend")
async def recommend(query: Query):
    try:
        analysis = enhance_query(query.query)
        logger.info(f"Query analysis: {json.dumps(analysis, indent=2)}")
        
        # Extract duration
        duration_matches = re.findall(r'\d+', query.query)
        max_duration = int(duration_matches[0]) if duration_matches else analysis.get("max_duration", 60)
        
        # Build search query
        search_terms = [
            analysis["role"],
            analysis["skills"],
            analysis["requirements"]
        ]
        search_query = " ".join(filter(None, search_terms))
        
        # Get embeddings
        query_embedding = embedder([search_query])
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=15,
            include=["metadatas", "distances"]
        )
        
        # Process analysis
        analysis_role = analysis.get("role", "").lower()
        analysis_skills = [s.strip().lower() for s in analysis.get("skills", "").split(",") if s.strip()]
        analysis_test_types = [tt.strip().lower() for tt in analysis.get("test_type", "").split(",") if tt.strip()]
        analysis_experience = [exp.strip().lower() for exp in analysis.get("experience", "").split(",") if exp.strip()]
        analysis_requirements = analysis.get("requirements", "").lower()

        recommendations = []
        for meta in results["metadatas"][0]:
            try:
                duration = int(meta["duration"])
                if duration > max_duration + 30:
                    continue

                # Extract metadata
                name = meta["name"].lower()
                desc = meta.get("description", "").lower()
                test_types = [tt.strip().lower() for tt in meta.get("test_types", "").split("|") if tt.strip()]
                job_levels = [jl.strip().lower() for jl in meta.get("job_levels", "").split("|") if jl.strip()]

                # Scoring
                score = 0
                
                # 1. High Priority: Role in name/description
                if analysis_role:
                    role_name = analysis_role in name
                    role_desc = analysis_role in desc
                    score += 3 * (role_name + role_desc)
                
                # 1. High Priority: Skills in name/description
                for skill in analysis_skills:
                    skill_name = skill in name
                    skill_desc = skill in desc
                    score += 2 * (skill_name + skill_desc)
                
                # 2. Test Type match
                matched_test_types = sum(1 for tt in analysis_test_types if tt in test_types)
                score += 2 * matched_test_types
                
                # 2. Experience match
                matched_experience = sum(1 for exp in analysis_experience if exp in job_levels)
                score += 2 * matched_experience
                
                # 3. Requirements in name/description
                if analysis_requirements:
                    req_name = analysis_requirements in name
                    req_desc = analysis_requirements in desc
                    score += 1 * (req_name + req_desc)
                
                # Duration proximity scoring
                duration_diff = abs(duration - max_duration)
                score += max(0, 3 - (duration_diff / 10))

                if score >= 5:
                    recommendations.append({
                        "assessment": Assessment(
                            name=meta["name"],
                            url=meta["url"],
                            duration=duration,
                            remote_support=meta["remote"],
                            adaptive_support=meta["adaptive"],
                            test_type=meta["test_types"].split("|")
                        ),
                        "score": score
                    })

            except Exception as e:
                logger.warning(f"Error processing metadata: {str(e)}")
                continue
        
        # Sort by score then duration
        recommendations.sort(key=lambda x: (-x["score"], x["assessment"].duration))
        
        return {"recommended_assessments": [rec["assessment"] for rec in recommendations[:10]]}
    
    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}")
        return {"error": "Internal server error"}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)