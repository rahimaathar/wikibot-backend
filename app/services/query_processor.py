import re
from typing import Dict, List
import nltk
import ssl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded"""
    try:
        # Handle SSL certificate verification for NLTK downloads
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Download required NLTK data
        required_data = [
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
        ]
        
        for path, package in required_data:
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"Downloading NLTK {package}...")
                nltk.download(package, quiet=True)
                print(f"Downloaded NLTK {package}")
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")

# Ensure NLTK data is available
ensure_nltk_data()

class QueryProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.question_types = {
            "what": "DEFINITION",
            "who": "FACTUAL",
            "when": "FACTUAL",
            "where": "FACTUAL",
            "why": "EXPLANATION",
            "how": "PROCESS",
            "compare": "COMPARISON",
            "difference": "COMPARISON",
            "explain": "EXPLANATION"
        }
        
        # Define response templates for different question types
        self.response_templates = {
            "DEFINITION": {
                "intro": "Here's what I found about {entity}:",
                "key_points": "Key points:",
                "context": "Additional context:"
            },
            "FACTUAL": {
                "intro": "Here's what I found:",
                "key_points": "Key points:",
                "context": "Additional information:"
            },
            "PROCESS": {
                "intro": "Here's how it works:",
                "key_points": "Key steps:",
                "context": "Important notes:"
            },
            "EXPLANATION": {
                "intro": "Here's what I found about {entity}:",
                "key_points": "Key points:",
                "context": "Additional context:"
            },
            "COMPARISON": {
                "intro": "Here's the comparison:",
                "key_points": "Key differences:",
                "context": "Additional context:"
            }
        }
    
    async def process(self, query: str) -> Dict:
        # Clean the query
        cleaned_query = query.lower().strip()
        
        # Determine question type
        question_type = self._determine_question_type(cleaned_query)
        
        # Extract entities
        entities = self._extract_entities(cleaned_query)
        
        # Generate alternative phrasings
        alternative_phrasings = self._generate_alternative_phrasings(cleaned_query, question_type)
        
        # Get response template for the question type
        response_template = self.response_templates.get(question_type, self.response_templates["EXPLANATION"])
        
        return {
            "original_query": query,
            "cleaned_query": cleaned_query,
            "question_type": question_type,
            "entities": entities,
            "alternative_phrasings": alternative_phrasings,
            "response_template": response_template
        }
    
    def _determine_question_type(self, query: str) -> str:
        """Determine the type of question being asked."""
        query = query.lower()
        
        # Define patterns for different question types
        patterns = {
            "DEFINITION": r"^(what is|define|explain|tell me about|describe|what are|what does)\s+",
            "FACTUAL": r"^(what|when|where|who|how|why)\s+",
            "PROCESS": r"^(how to|how do|steps to|process of|how does|how can)\s+",
            "EXPLANATION": r"^(why|explain why|reason for|what causes|what makes|let me explain|explain)\s+",
            "COMPARISON": r"^(compare|difference between|versus|vs|similarities between|differences between)\s+"
        }
        
        # Check each pattern
        for q_type, pattern in patterns.items():
            if re.match(pattern, query):
                return q_type
        
        # Default to explanation if no pattern matches
        return "EXPLANATION"
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract main entities from the query."""
        # Tokenize and tag parts of speech
        tokens = word_tokenize(query.lower())
        tagged = pos_tag(tokens)
        
        # Extract nouns and named entities
        entities = []
        for word, tag in tagged:
            if tag.startswith('NN') and word not in self.stop_words and len(word) > 2:
                entities.append(word)
        
        return entities
    
    def _generate_alternative_phrasings(self, query: str, question_type: str) -> List[str]:
        """Generate alternative ways to phrase the query."""
        phrasings = []
        
        # Remove question words and common prefixes
        base_query = re.sub(r'^(what is|define|explain|tell me about|how to|how do|steps to|process of|why|explain why|reason for|compare|difference between|versus|vs|let me explain)\s+', '', query)
        
        # Generate variations based on question type
        if question_type == "DEFINITION":
            phrasings.extend([
                f"what is {base_query}",
                f"define {base_query}",
                f"explain {base_query}",
                f"describe {base_query}",
                f"what are the characteristics of {base_query}"
            ])
        elif question_type == "FACTUAL":
            phrasings.extend([
                f"what is {base_query}",
                f"information about {base_query}",
                f"details about {base_query}",
                f"tell me about {base_query}",
                f"what are the facts about {base_query}"
            ])
        elif question_type == "PROCESS":
            phrasings.extend([
                f"how to {base_query}",
                f"steps to {base_query}",
                f"process of {base_query}",
                f"how does {base_query} work",
                f"what is the process of {base_query}"
            ])
        elif question_type == "EXPLANATION":
            phrasings.extend([
                f"explain {base_query}",
                f"tell me about {base_query}",
                f"what is {base_query}",
                f"describe {base_query}",
                f"what are the key aspects of {base_query}"
            ])
        elif question_type == "COMPARISON":
            phrasings.extend([
                f"compare {base_query}",
                f"differences between {base_query}",
                f"similarities between {base_query}",
                f"how does {base_query} compare",
                f"what are the differences in {base_query}"
            ])
        
        return phrasings

# Create a singleton instance
query_processor = QueryProcessor()

async def process_query(query: str) -> Dict:
    return await query_processor.process(query) 