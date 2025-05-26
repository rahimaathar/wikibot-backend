import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self):
        self.response_templates = {
            "DEFINITION": "{intro}\n\n{main_content}",
            "FACTUAL": "{intro}\n\n{main_content}",
            "PROCESS": "{intro}\n\n{main_content}",
            "EXPLANATION": "{intro}\n\n{main_content}",
            "COMPARISON": "{intro}\n\n{main_content}"
        }
        
    async def generate(self, analyzed_content: Dict, processed_query: Dict) -> Dict:
        """Generate a response based on analyzed content and query type"""
        try:
            if not analyzed_content or not analyzed_content.get("main_content"):
                return {
                    "response": "I apologize, but I couldn't find enough relevant information to answer your question.",
                    "confidence": 0.0,
                    "sources": []
                }
            
            # Get the main content and confidence
            content = analyzed_content["main_content"]
            confidence = analyzed_content["confidence"]
            
            # Get the query type and topic
            query_type = processed_query.get("question_type", "EXPLANATION")
            topic = processed_query.get("cleaned_query", "this topic")
            
            # Format the response using the appropriate template
            template = self.response_templates.get(query_type, self.response_templates["EXPLANATION"])
            
            # Create a focused response
            intro = f"Let me explain {topic}:"
            
            response = template.format(
                intro=intro,
                main_content=content
            )
            
            return {
                "response": response,
                "confidence": confidence,
                "sources": analyzed_content.get("sources", [])
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {
                "response": "I apologize, but I encountered an error while processing your query. Please try again.",
                "confidence": 0.0,
                "sources": []
            }
    
    def _format_main_content(self, content: str, key_terms: List[str]) -> str:
        """Format the main content into a focused, coherent explanation"""
        # Split content into sentences
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        if not sentences:
            return content
            
        # Group related sentences together
        paragraphs = []
        current_para = []
        current_topic = None
        
        for sentence in sentences:
            # If this is a new topic or we have enough sentences for a paragraph
            if len(current_para) >= 3 or (current_topic and not self._is_same_topic(sentence, current_topic)):
                if current_para:
                    para_text = '. '.join(current_para) + '.'
                    paragraphs.append(para_text)
                current_para = [sentence]
                current_topic = sentence
            else:
                current_para.append(sentence)
                if not current_topic:
                    current_topic = sentence
        
        # Add the last paragraph if it exists
        if current_para:
            para_text = '. '.join(current_para) + '.'
            paragraphs.append(para_text)
        
        # Return the most relevant paragraph if we have multiple
        if len(paragraphs) > 1:
            # Choose the paragraph with the most key terms
            key_terms_set = set(term.lower() for term in key_terms)
            best_para = max(paragraphs, key=lambda p: sum(1 for term in key_terms_set if term in p.lower()))
            return best_para
        
        return '\n\n'.join(paragraphs)
    
    def _is_same_topic(self, sentence: str, topic: str) -> bool:
        """Check if a sentence is about the same topic as the current paragraph"""
        # Simple heuristic: check for common words
        topic_words = set(word.lower() for word in topic.split() if len(word) > 3)
        sentence_words = set(word.lower() for word in sentence.split() if len(word) > 3)
        return bool(topic_words & sentence_words)

# Create singleton instance
response_generator = ResponseGenerator()

async def generate_response(analyzed_content: Dict, processed_query: Dict) -> Dict:
    """Public function to generate response"""
    return await response_generator.generate(analyzed_content, processed_query)
