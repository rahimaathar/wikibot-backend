from transformers import pipeline
from typing import Dict, List, Optional
import re

class QAModel:
    def __init__(self):
        # Initialize the QA pipeline with a pre-trained model
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2"
        )
    
    def extract_answer(self, question: str, context: str) -> Dict:
        """Extract answer from context using the QA model."""
        try:
            # Clean and prepare the context
            context = self._clean_context(context)
            
            # Get answer from QA model
            result = self.qa_pipeline(
                question=question,
                context=context,
                max_answer_len=100,
                handle_impossible_answer=True
            )
            
            return {
                "answer": result["answer"],
                "confidence": result["score"],
                "start": result["start"],
                "end": result["end"]
            }
        except Exception as e:
            print(f"Error in QA extraction: {str(e)}")
            return {
                "answer": "",
                "confidence": 0.0,
                "start": 0,
                "end": 0
            }
    
    def _clean_context(self, context: str) -> str:
        """Clean and prepare context for QA model."""
        # Remove extra whitespace
        context = re.sub(r'\s+', ' ', context)
        # Remove special characters
        context = re.sub(r'[^\w\s.,!?-]', '', context)
        return context.strip()
    
    def extract_comparison(self, query: str, contexts: List[str]) -> Dict:
        """Extract comparison information from multiple contexts."""
        try:
            # Extract key terms from query
            terms = self._extract_comparison_terms(query)
            if not terms or len(terms) < 2:
                return {
                    "answer": "Could not identify terms to compare.",
                    "confidence": 0.0
                }
            
            # Get answers for each term
            answers = []
            for term in terms:
                for context in contexts:
                    answer = self.extract_answer(f"What is {term}?", context)
                    if answer["confidence"] > 0.5:
                        answers.append({
                            "term": term,
                            "answer": answer["answer"],
                            "confidence": answer["confidence"]
                        })
            
            if not answers:
                return {
                    "answer": "Could not find sufficient information for comparison.",
                    "confidence": 0.0
                }
            
            # Format comparison
            comparison = self._format_comparison(answers)
            return {
                "answer": comparison,
                "confidence": min(a["confidence"] for a in answers)
            }
        except Exception as e:
            print(f"Error in comparison extraction: {str(e)}")
            return {
                "answer": "Error processing comparison request.",
                "confidence": 0.0
            }
    
    def _extract_comparison_terms(self, query: str) -> List[str]:
        """Extract terms to compare from the query."""
        # Look for comparison patterns
        patterns = [
            r"compare\s+(\w+)\s+and\s+(\w+)",
            r"difference\s+between\s+(\w+)\s+and\s+(\w+)",
            r"(\w+)\s+versus\s+(\w+)",
            r"(\w+)\s+vs\s+(\w+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return [match.group(1), match.group(2)]
        
        return []
    
    def _format_comparison(self, answers: List[Dict]) -> str:
        """Format comparison answers into a coherent response."""
        if not answers:
            return "No comparison information found."
        
        # Group answers by term
        term_answers = {}
        for answer in answers:
            term = answer["term"]
            if term not in term_answers:
                term_answers[term] = []
            term_answers[term].append(answer)
        
        # Format the comparison
        comparison = []
        for term, term_info in term_answers.items():
            # Get the best answer for each term
            best_answer = max(term_info, key=lambda x: x["confidence"])
            comparison.append(f"{term.title()}: {best_answer['answer']}")
        
        return "\n\n".join(comparison)

# Create a singleton instance
qa_model = QAModel() 