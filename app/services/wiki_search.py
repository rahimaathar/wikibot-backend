import wikipediaapi
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from datetime import timedelta
from cachetools import TTLCache
import json
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikipediaSearch:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='WikiChatBot/1.0 (https://en.wikipedia.org/wiki/Main_Page; bot@example.com) Python/3.12',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.cache = TTLCache(maxsize=1000, ttl=timedelta(hours=1).total_seconds())

    def _get_cache_key(self, query: Dict) -> str:
        return json.dumps(query, sort_keys=True)

    async def search(self, processed_query: Dict) -> List[Dict]:
        try:
            search_results = self._perform_search(processed_query)
            if not search_results:
                logger.warning("No search results found for query")
                return []

            detailed_results = await self._get_article_content(search_results[:3], processed_query)
            if not detailed_results:
                logger.warning("No detailed content found for results")
                return []

            return self._process_results(detailed_results, processed_query)

        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            return []

    def _perform_search(self, processed_query: Dict) -> List[Dict]:
        search_terms = [processed_query["cleaned_query"]] + processed_query.get("alternative_phrasings", [])
        all_results = []

        for term in search_terms:
            try:
                clean_term = self._clean_search_term(term)
                page = self.wiki.page(clean_term)
                if page.exists() and not self._is_disambiguation(page):
                    all_results.append({
                        "title": page.title,
                        "pageid": page.pageid,
                        "relevance": 1.0
                    })
                    continue

                # Wikipedia API fallback
                search_variations = [
                    clean_term, clean_term.replace(" ", "_"),
                    clean_term.title(), clean_term.lower()
                ]

                for search_term in search_variations:
                    params = {
                        "action": "query",
                        "list": "search",
                        "srsearch": search_term,
                        "format": "json",
                        "srlimit": 5
                    }
                    response = requests.get("https://en.wikipedia.org/w/api.php", params=params)
                    data = response.json()

                    if "query" in data and "search" in data["query"]:
                        for idx, result in enumerate(data["query"]["search"]):
                            page = self.wiki.page(result["title"])
                            if page.exists() and not self._is_disambiguation(page):
                                all_results.append({
                                    "title": page.title,
                                    "pageid": page.pageid,
                                    "relevance": 0.9 - (idx * 0.1)
                                })
            except Exception as e:
                logger.warning(f"Search failed for term '{term}': {str(e)}")

        return self._deduplicate_and_sort(all_results)

    def _clean_search_term(self, term: str) -> str:
        # Remove punctuation and common prefixes
        term = re.sub(r'[?.,!]', '', term)
        term = re.sub(r'^(what is|who is|define|explain|how did|why did|let me explain)\s+', '', term, flags=re.IGNORECASE)
        # Remove extra spaces and trim
        term = re.sub(r'\s+', ' ', term).strip()
        # Capitalize first letter of each word for better matching
        return ' '.join(word.capitalize() for word in term.split())

    def _is_disambiguation(self, page) -> bool:
        return "(disambiguation)" in page.title or "may refer to:" in page.summary[:100]

    async def _get_article_content(self, results: List[Dict], processed_query: Dict) -> List[Dict]:
        detailed_results = []

        for result in results:
            try:
                page = self.wiki.page(result["title"])
                if not page.exists():
                    continue

                content = page.summary or ""

                if page.sections:
                    for section in page.sections:
                        if any(skip in section.title.lower() for skip in ["see also", "references", "external links", "notes"]):
                            continue
                        if section.text:
                            section_text = re.sub(r'\[\d+\]', '', section.text)
                            section_text = re.sub(r'\s+', ' ', section_text)
                            content += "\n\n" + section_text

                if len(content.split()) < 500 and page.text:
                    full_text = re.sub(r'\[\d+\]', '', page.text)
                    full_text = re.sub(r'==.*?==', '', full_text)
                    full_text = re.sub(r'\s+', ' ', full_text)
                    content += "\n\n" + full_text

                if content:
                    detailed_results.append({
                        **result,
                        "content": content,
                        "url": page.fullurl
                    })

            except Exception as e:
                logger.warning(f"Failed to get content for {result.get('title')}: {str(e)}")

        return detailed_results

    def _process_results(self, results: List[Dict], processed_query: Dict) -> List[Dict]:
        processed_results = []

        for result in results:
            try:
                relevance = self._calculate_relevance(result["content"], processed_query["original_query"])
                if relevance >= 0.3:
                    content = self._extract_relevant_content(result["content"], processed_query["original_query"])
                    processed_results.append({
                        **result,
                        "content": content,
                        "confidence": relevance
                    })
            except Exception as e:
                logger.warning(f"Failed to process result {result.get('title')}: {str(e)}")

        return sorted(processed_results, key=lambda x: x["confidence"], reverse=True)

    def _calculate_relevance(self, content: str, query: str) -> float:
        try:
            # Clean and normalize content
            content = re.sub(r'\s+', ' ', content).strip()
            query = re.sub(r'\s+', ' ', query).strip()
            
            # Calculate TF-IDF similarity
            tfidf_matrix = self.vectorizer.fit_transform([query, content])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Calculate length factor
            content_length = len(content.split())
            length_factor = min(1.0, content_length / 200)
            
            # Boost similarity if query terms are present
            query_terms = set(query.lower().split())
            content_lower = content.lower()
            term_matches = sum(1 for term in query_terms if term in content_lower)
            term_boost = 1.0 + (term_matches * 0.2)  # Increased boost
            
            # Calculate final relevance with higher weight on term matches
            relevance = (similarity * 0.4) + (length_factor * 0.3) + (term_boost * 0.3)
            
            return min(1.0, relevance)
        except Exception as e:
            logger.warning(f"Failed to calculate relevance: {str(e)}")
            return 0.0

    def _extract_relevant_content(self, content: str, query: str) -> str:
        try:
            # Split into paragraphs first
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            if not paragraphs:
                return content

            # Calculate relevance for each paragraph
            tfidf_matrix = self.vectorizer.fit_transform([query] + paragraphs)
            query_vec = tfidf_matrix[0:1]
            paragraph_vecs = tfidf_matrix[1:]

            similarities = cosine_similarity(query_vec, paragraph_vecs).flatten()
            
            # Get top 3 most relevant paragraphs
            top_indices = np.argsort(similarities)[-3:][::-1]
            top_paragraphs = [paragraphs[i] for i in top_indices]

            # If paragraphs are too short, try sentence-level extraction
            if sum(len(p.split()) for p in top_paragraphs) < 100:
                sentences = re.split(r'[.!?]+', content)
                sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 5]

                if sentences:
                    tfidf_matrix = self.vectorizer.fit_transform([query] + sentences)
                    query_vec = tfidf_matrix[0:1]
                    sentence_vecs = tfidf_matrix[1:]

                    similarities = cosine_similarity(query_vec, sentence_vecs).flatten()
                    top_indices = np.argsort(similarities)[-5:][::-1]
                    top_sentences = [sentences[i] for i in top_indices]
                    return '. '.join(top_sentences).strip() + '.'

            return '\n\n'.join(top_paragraphs)
        except Exception as e:
            logger.warning(f"Failed to extract relevant content: {str(e)}")
            return content

    def _deduplicate_and_sort(self, results: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for result in sorted(results, key=lambda x: -x["relevance"]):
            if result["title"] not in seen:
                seen.add(result["title"])
                unique.append(result)
        return unique

# Create a singleton instance
wiki_search = WikipediaSearch()

async def search_wikipedia(processed_query: Dict) -> List[Dict]:
    """Search Wikipedia using the processed query."""
    return await wiki_search.search(processed_query)
