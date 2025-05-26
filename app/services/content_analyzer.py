import re
import logging
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentAnalyzer:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        # Define section patterns for different topic types
        self.section_patterns = {
            "GEOGRAPHY": {
                "Overview": ['overview', 'introduction', 'about', 'general', 'basic'],
                "Location": ['located', 'situated', 'found in', 'position', 'coordinates'],
                "Physical Features": ['mountain', 'river', 'lake', 'climate', 'terrain', 'landscape'],
                "History": ['history', 'historical', 'founded', 'established', 'discovered'],
                "Culture": ['culture', 'people', 'population', 'language', 'traditions'],
                "Information": ['information', 'details', 'facts', 'data', 'statistics']
            },
            "SCIENCE": {
                "Overview": ['overview', 'introduction', 'about', 'general', 'basic'],
                "Definition": ['defined as', 'refers to', 'means', 'is a', 'consists of'],
                "Process": ['process', 'how it works', 'mechanism', 'steps', 'stages'],
                "Applications": ['used in', 'applied to', 'applications', 'uses', 'practical'],
                "Research": ['studies', 'research', 'experiments', 'findings', 'discoveries'],
                "Information": ['information', 'details', 'facts', 'data', 'statistics']
            },
            "HISTORY": {
                "Overview": ['overview', 'introduction', 'about', 'general', 'basic'],
                "Background": ['background', 'context', 'situation', 'circumstances'],
                "Events": ['event', 'occurred', 'happened', 'took place', 'began'],
                "Impact": ['impact', 'effect', 'influence', 'consequences', 'resulted in'],
                "Significance": ['significant', 'important', 'notable', 'memorable', 'historic'],
                "Information": ['information', 'details', 'facts', 'data', 'statistics']
            },
            "TOOL": {
                "Overview": ['overview', 'introduction', 'about', 'general', 'basic'],
                "Definition": ['defined as', 'refers to', 'means', 'is a', 'consists of', 'tool', 'device', 'instrument'],
                "History": ['history', 'historical', 'origin', 'developed', 'invented', 'created', 'first used'],
                "Usage": ['how to use', 'operation', 'function', 'purpose', 'used for', 'application'],
                "Components": ['parts', 'components', 'structure', 'made of', 'constructed', 'design'],
                "Benefits": ['benefits', 'advantages', 'importance', 'significance', 'value', 'useful'],
                "Information": ['information', 'details', 'facts', 'data', 'statistics']
            },
            "GENERAL": {
                "Overview": ['overview', 'introduction', 'about', 'general', 'basic'],
                "Details": ['details', 'specifics', 'particulars', 'characteristics', 'features'],
                "Examples": ['example', 'instance', 'case', 'illustration', 'sample'],
                "Related": ['related', 'connected', 'associated', 'similar', 'relevant'],
                "Information": ['information', 'details', 'facts', 'data', 'statistics']
            }
        }

    def _clean_text(self, text: str) -> str:
        """Clean and format text."""
        # Remove citations and references
        text = re.sub(r'\[\d+\]', '', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing punctuation
        text = text.strip('.,;: ')
        return text

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [self._clean_text(s) for s in sentences if s.strip()]
        
        # Filter out very short sentences and ensure they're complete
        key_points = []
        for sentence in sentences:
            if len(sentence.split()) > 5 and sentence.endswith(('.', '!', '?')):
                # Remove common prefixes that make sentences less direct
                sentence = re.sub(r'^(it is|this is|there is|there are|they are|we can|you can|one can)\s+', '', sentence, flags=re.IGNORECASE)
                key_points.append(sentence)
        
        return key_points

    def _determine_topic_type(self, content: str, main_topic: str) -> str:
        """Determine the type of topic based on content and main topic."""
        content_lower = content.lower()
        topic_lower = main_topic.lower()
        
        # Check for tool/device indicators
        tool_terms = [
            'tool', 'device', 'instrument', 'machine', 'apparatus',
            'calculator', 'abacus', 'compass', 'ruler', 'protractor',
            'used for', 'purpose', 'function', 'operation'
        ]
        if any(term in topic_lower for term in tool_terms) or any(term in content_lower for term in tool_terms):
            return "TOOL"
            
        # Check for scientific/biological indicators
        science_terms = [
            'species', 'genus', 'family', 'animal', 'plant', 'organism',
            'theory', 'process', 'phenomenon', 'principle', 'law', 'concept',
            'biology', 'chemistry', 'physics', 'science', 'scientific'
        ]
        if any(term in topic_lower for term in science_terms) or any(term in content_lower for term in science_terms):
            return "SCIENCE"
            
        # Check for geographical indicators
        geo_terms = ['country', 'city', 'mountain', 'river', 'lake', 'ocean', 'continent']
        if any(term in topic_lower for term in geo_terms) or any(term in content_lower for term in geo_terms):
            return "GEOGRAPHY"
            
        # Check for historical events
        history_terms = ['war', 'battle', 'period', 'era', 'century', 'dynasty', 'revolution']
        if any(term in topic_lower for term in history_terms) or any(term in content_lower for term in history_terms):
            return "HISTORY"
            
        # Default to general
        return "GENERAL"

    def _structure_content(self, content: str, main_topic: str) -> str:
        """Structure content into sections based on topic type."""
        if not content:
            return ""

        # Determine topic type
        topic_type = self._determine_topic_type(content, main_topic)
        section_patterns = self.section_patterns[topic_type]
        
        # Extract key points
        key_points = self._extract_key_points(content)
        
        if not key_points:
            # If no key points found, try to use the content directly
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            key_points = [s for s in sentences if len(s.split()) > 5][:5]
        
        # Initialize sections
        sections = {title: [] for title in section_patterns.keys()}
        
        # Categorize sentences into sections
        for point in key_points:
            point_lower = point.lower()
            categorized = False
            
            for section, patterns in section_patterns.items():
                if any(pattern in point_lower for pattern in patterns):
                    sections[section].append(point)
                    categorized = True
                    break
            
            # If point doesn't fit any section, add to Information
            if not categorized:
                sections["Information"].append(point)

        # Build the response
        result = []
        
        # Add sections that have content
        for title, points in sections.items():
            if points:
                result.append(f"\n{title}:")
                # Take up to 5 points per section
                for point in points[:5]:
                    result.append(f"• {point}")

        # If no sections have content, return a basic structure
        if not result:
            result.append("\nInformation:")
            for point in key_points[:5]:
                result.append(f"• {point}")

        return "\n".join(result)

    def analyze_content(self, documents: List[Document], main_topic: str) -> Dict:
        try:
            # Merge documents
            content = " ".join(doc.page_content for doc in documents if doc.page_content)
            
            if not content:
                return {
                    "main_content": "I couldn't find any relevant information about this topic.",
                    "confidence": 0.0,
                    "key_terms": [],
                    "sources": []
                }
            
            # Structure the content
            structured_content = self._structure_content(content, main_topic)
            
            # Calculate confidence based on content length and quality
            word_count = len(structured_content.split())
            confidence = min(1.0, word_count / 500)
            
            # Format sources
            sources = []
            for doc in documents:
                if hasattr(doc, 'metadata'):
                    title = doc.metadata.get('title', '')
                    url = doc.metadata.get('url', '')
                    if title and url:
                        sources.append(f"{title} - {url}")
            
            # Ensure we have a valid response
            if not structured_content or structured_content.isspace():
                structured_content = "I found some information about this topic, but couldn't structure it properly. Here's what I found:\n\n" + content[:500] + "..."
            
            return {
                "main_content": structured_content,
                "confidence": confidence,
                "key_terms": [],
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Error in analyze_content: {str(e)}", exc_info=True)
            return {
                "main_content": "I encountered an error while processing the information.",
                "confidence": 0.0,
                "key_terms": [],
                "sources": []
            }

# Create a singleton instance
content_analyzer = ContentAnalyzer()

async def analyze_content(search_results: List[Dict], processed_query: Dict) -> Dict:
    """Analyze content from search results and structure it based on the main topic."""
    try:
        if not search_results:
            return {
                "main_content": "I couldn't find any relevant information about this topic.",
                "confidence": 0.0,
                "key_terms": [],
                "sources": []
            }
        
        # Convert search results to Document objects
        documents = []
        for result in search_results:
            if not isinstance(result, dict):
                continue
            content = result.get("content", "")
            if not content:
                continue
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "title": result.get("title", ""),
                        "url": result.get("url", "")
                    }
                )
            )
        
        if not documents:
            return {
                "main_content": "I couldn't find any relevant information about this topic.",
                "confidence": 0.0,
                "key_terms": [],
                "sources": []
            }
        
        # Get the main topic from the processed query, safely handling empty entities list
        entities = processed_query.get("entities", [])
        main_topic = entities[0] if entities else processed_query.get("cleaned_query", "")
        
        return content_analyzer.analyze_content(documents, main_topic)
        
    except Exception as e:
        logger.error(f"Error analyzing content: {str(e)}", exc_info=True)
        return {
            "main_content": "I encountered an error while processing the information.",
            "confidence": 0.0,
            "key_terms": [],
            "sources": []
        }
