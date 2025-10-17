"""
Citation Monitor Module

Monitors academic citations and research references to the RiskX platform.
Tracks scholarly impact and academic validation across multiple databases.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp
import xml.etree.ElementTree as ET
from urllib.parse import quote

from ..core.config import get_settings
from ..cache.cache_manager import CacheManager


@dataclass
class Citation:
    """Represents an academic citation"""
    citation_id: str
    title: str
    authors: List[str]
    journal: str
    publication_date: datetime
    abstract: str
    doi: Optional[str]
    url: str
    citation_type: str  # 'direct', 'indirect', 'methodology', 'data'
    relevance_score: float
    keywords_matched: List[str]
    citation_context: str  # How RiskX was referenced


@dataclass
class CitationMetrics:
    """Aggregated citation metrics"""
    total_citations: int
    direct_citations: int
    indirect_citations: int
    methodology_references: int
    data_usage_citations: int
    top_citing_journals: List[str]
    citation_growth_rate: float
    h_index_estimate: int
    recent_citations: List[Citation]


class CitationMonitor:
    """
    Monitors academic citations and scholarly references to RiskX platform.
    
    Tracks citations across:
    - ArXiv preprints
    - SSRN working papers  
    - Google Scholar results
    - PubMed database
    - RePEc economics papers
    - Government research reports
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = CacheManager()
        self.logger = logging.getLogger(__name__)
        
        # Search terms to monitor for citations
        self.search_terms = [
            "RiskX", "Risk Intelligence Observatory", "AI Risk Intelligence",
            "systemic risk prediction", "explainable risk assessment",
            "supply chain risk intelligence", "financial stability AI"
        ]
        
        # Platforms to monitor
        self.citation_sources = {
            "arxiv": "http://export.arxiv.org/api/query",
            "ssrn": "https://api.ssrn.com/search",  # Note: May need API key
            "pubmed": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            "crossref": "https://api.crossref.org/works"
        }
    
    async def check_new_citations(self) -> List[Citation]:
        """
        Check for new citations across all monitored platforms
        """
        self.logger.info("Starting citation monitoring")
        
        try:
            all_citations = []
            
            # Check each citation source
            for source_name, base_url in self.citation_sources.items():
                try:
                    citations = await self._search_citations(source_name, base_url)
                    all_citations.extend(citations)
                    self.logger.info(f"Found {len(citations)} citations from {source_name}")
                except Exception as e:
                    self.logger.warning(f"Error searching {source_name}: {str(e)}")
            
            # Filter for new citations
            new_citations = await self._filter_new_citations(all_citations)
            
            # Cache new citations
            if new_citations:
                await self._cache_citations(new_citations)
            
            self.logger.info(f"Citation monitoring completed. Found {len(new_citations)} new citations")
            return new_citations
            
        except Exception as e:
            self.logger.error(f"Error in citation monitoring: {str(e)}")
            return []
    
    async def _search_citations(self, source_name: str, base_url: str) -> List[Citation]:
        """Search for citations from a specific source"""
        citations = []
        
        async with aiohttp.ClientSession() as session:
            for search_term in self.search_terms:
                try:
                    if source_name == "arxiv":
                        source_citations = await self._search_arxiv(session, base_url, search_term)
                    elif source_name == "pubmed":
                        source_citations = await self._search_pubmed(session, base_url, search_term)
                    elif source_name == "crossref":
                        source_citations = await self._search_crossref(session, base_url, search_term)
                    else:
                        # Generic search for other sources
                        source_citations = await self._generic_search(session, base_url, search_term, source_name)
                    
                    citations.extend(source_citations)
                    
                    # Rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.warning(f"Error searching {source_name} for '{search_term}': {str(e)}")
        
        return citations
    
    async def _search_arxiv(self, session: aiohttp.ClientSession, base_url: str, search_term: str) -> List[Citation]:
        """Search ArXiv for citations"""
        citations = []
        
        # Construct ArXiv query
        query = quote(f'all:"{search_term}"')
        url = f"{base_url}?search_query={query}&start=0&max_results=20"
        
        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    content = await response.text()
                    citations = self._parse_arxiv_response(content, search_term)
        except Exception as e:
            self.logger.error(f"Error querying ArXiv: {str(e)}")
        
        return citations
    
    async def _search_pubmed(self, session: aiohttp.ClientSession, base_url: str, search_term: str) -> List[Citation]:
        """Search PubMed for citations"""
        citations = []
        
        # Construct PubMed query
        params = {
            "db": "pubmed",
            "term": search_term,
            "retmax": "20",
            "retmode": "xml"
        }
        
        try:
            async with session.get(base_url, params=params, timeout=30) as response:
                if response.status == 200:
                    content = await response.text()
                    citations = self._parse_pubmed_response(content, search_term)
        except Exception as e:
            self.logger.error(f"Error querying PubMed: {str(e)}")
        
        return citations
    
    async def _search_crossref(self, session: aiohttp.ClientSession, base_url: str, search_term: str) -> List[Citation]:
        """Search Crossref for citations"""
        citations = []
        
        # Construct Crossref query
        params = {
            "query": search_term,
            "rows": "20",
            "sort": "published",
            "order": "desc"
        }
        
        try:
            async with session.get(base_url, params=params, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    citations = self._parse_crossref_response(data, search_term)
        except Exception as e:
            self.logger.error(f"Error querying Crossref: {str(e)}")
        
        return citations
    
    async def _generic_search(self, session: aiohttp.ClientSession, base_url: str, search_term: str, source_name: str) -> List[Citation]:
        """Generic search for sources without specific implementation"""
        # Placeholder for future source implementations
        self.logger.info(f"Generic search not implemented for {source_name}")
        return []
    
    def _parse_arxiv_response(self, xml_content: str, search_term: str) -> List[Citation]:
        """Parse ArXiv XML response"""
        citations = []
        
        try:
            root = ET.fromstring(xml_content)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            
            for entry in root.findall("atom:entry", namespace):
                try:
                    title = entry.find("atom:title", namespace)
                    title_text = title.text.strip() if title is not None else ""
                    
                    # Extract authors
                    authors = []
                    for author in entry.findall("atom:author", namespace):
                        name = author.find("atom:name", namespace)
                        if name is not None:
                            authors.append(name.text.strip())
                    
                    # Extract abstract
                    summary = entry.find("atom:summary", namespace)
                    abstract = summary.text.strip() if summary is not None else ""
                    
                    # Extract URL
                    link = entry.find("atom:link[@type='text/html']", namespace)
                    url = link.get("href") if link is not None else ""
                    
                    # Extract publication date
                    published = entry.find("atom:published", namespace)
                    pub_date = datetime.now()
                    if published is not None:
                        try:
                            pub_date = datetime.fromisoformat(published.text.replace('Z', '+00:00'))
                        except:
                            pass
                    
                    # Calculate relevance and extract citation context
                    relevance_score, keywords_matched, context = self._analyze_citation_relevance(
                        title_text, abstract, search_term
                    )
                    
                    if relevance_score > 0.3:  # Minimum relevance threshold
                        citation = Citation(
                            citation_id=f"arxiv_{url.split('/')[-1] if url else 'unknown'}",
                            title=title_text,
                            authors=authors,
                            journal="arXiv",
                            publication_date=pub_date,
                            abstract=abstract[:500],  # Truncate
                            doi=None,
                            url=url,
                            citation_type=self._determine_citation_type(context),
                            relevance_score=relevance_score,
                            keywords_matched=keywords_matched,
                            citation_context=context
                        )
                        citations.append(citation)
                
                except Exception as e:
                    self.logger.warning(f"Error parsing ArXiv entry: {str(e)}")
        
        except ET.ParseError as e:
            self.logger.error(f"Error parsing ArXiv XML: {str(e)}")
        
        return citations
    
    def _parse_pubmed_response(self, xml_content: str, search_term: str) -> List[Citation]:
        """Parse PubMed XML response"""
        citations = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # PubMed search returns ID list, would need additional API call
            # for full details. This is a simplified implementation.
            id_list = root.find("IdList")
            if id_list is not None:
                ids = [id_elem.text for id_elem in id_list.findall("Id")]
                # In a full implementation, would fetch details for each ID
                self.logger.info(f"Found {len(ids)} PubMed results for '{search_term}'")
        
        except ET.ParseError as e:
            self.logger.error(f"Error parsing PubMed XML: {str(e)}")
        
        return citations
    
    def _parse_crossref_response(self, data: Dict, search_term: str) -> List[Citation]:
        """Parse Crossref JSON response"""
        citations = []
        
        try:
            items = data.get("message", {}).get("items", [])
            
            for item in items:
                try:
                    title = item.get("title", [""])[0]
                    
                    # Extract authors
                    authors = []
                    for author in item.get("author", []):
                        given = author.get("given", "")
                        family = author.get("family", "")
                        if given and family:
                            authors.append(f"{given} {family}")
                        elif family:
                            authors.append(family)
                    
                    # Extract journal
                    journal = item.get("container-title", ["Unknown"])[0]
                    
                    # Extract DOI and URL
                    doi = item.get("DOI")
                    url = item.get("URL", f"https://doi.org/{doi}" if doi else "")
                    
                    # Extract publication date
                    pub_date = datetime.now()
                    date_parts = item.get("published-print", {}).get("date-parts")
                    if date_parts and len(date_parts[0]) >= 3:
                        try:
                            year, month, day = date_parts[0][:3]
                            pub_date = datetime(year, month, day)
                        except:
                            pass
                    
                    # Abstract not always available in Crossref
                    abstract = item.get("abstract", "")
                    
                    # Calculate relevance
                    relevance_score, keywords_matched, context = self._analyze_citation_relevance(
                        title, abstract, search_term
                    )
                    
                    if relevance_score > 0.3:
                        citation = Citation(
                            citation_id=f"crossref_{doi}" if doi else f"crossref_{hash(title)}",
                            title=title,
                            authors=authors,
                            journal=journal,
                            publication_date=pub_date,
                            abstract=abstract[:500],
                            doi=doi,
                            url=url,
                            citation_type=self._determine_citation_type(context),
                            relevance_score=relevance_score,
                            keywords_matched=keywords_matched,
                            citation_context=context
                        )
                        citations.append(citation)
                
                except Exception as e:
                    self.logger.warning(f"Error parsing Crossref item: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error parsing Crossref response: {str(e)}")
        
        return citations
    
    def _analyze_citation_relevance(self, title: str, abstract: str, search_term: str) -> tuple[float, List[str], str]:
        """Analyze citation relevance and extract context"""
        text = f"{title} {abstract}".lower()
        
        # Keywords that indicate RiskX citation
        risk_keywords = [
            "risk intelligence", "systemic risk", "supply chain risk",
            "financial stability", "risk assessment", "predictive analytics",
            "explainable ai", "risk prediction", "economic vulnerability"
        ]
        
        matched_keywords = []
        score = 0.0
        context = ""
        
        # Check for direct mentions
        if search_term.lower() in text:
            score += 0.5
            context = "direct_mention"
        
        # Check for related keywords
        for keyword in risk_keywords:
            if keyword in text:
                matched_keywords.append(keyword)
                score += 0.1
        
        # Check for methodology references
        methodology_terms = ["methodology", "framework", "approach", "model", "algorithm"]
        if any(term in text for term in methodology_terms) and matched_keywords:
            score += 0.2
            if not context:
                context = "methodology_reference"
        
        # Check for data usage
        data_terms = ["dataset", "data source", "benchmark", "evaluation"]
        if any(term in text for term in data_terms) and matched_keywords:
            score += 0.2
            if not context:
                context = "data_usage"
        
        # Normalize score
        score = min(score, 1.0)
        
        return score, matched_keywords, context or "general_reference"
    
    def _determine_citation_type(self, context: str) -> str:
        """Determine the type of citation based on context"""
        context_mapping = {
            "direct_mention": "direct",
            "methodology_reference": "methodology", 
            "data_usage": "data",
            "general_reference": "indirect"
        }
        return context_mapping.get(context, "indirect")
    
    async def _filter_new_citations(self, citations: List[Citation]) -> List[Citation]:
        """Filter out citations that have already been tracked"""
        if not citations:
            return []
        
        try:
            # Get existing citation IDs from cache
            existing_ids = await self.cache.get("tracked_citation_ids") or set()
            
            # Filter for new citations
            new_citations = []
            new_ids = set()
            
            for citation in citations:
                if citation.citation_id not in existing_ids:
                    new_citations.append(citation)
                    new_ids.add(citation.citation_id)
            
            # Update tracked IDs
            if new_ids:
                updated_ids = existing_ids.union(new_ids)
                await self.cache.set("tracked_citation_ids", list(updated_ids), ttl=86400 * 30)
            
            return new_citations
            
        except Exception as e:
            self.logger.error(f"Error filtering new citations: {str(e)}")
            return citations  # Return all if filtering fails
    
    async def _cache_citations(self, citations: List[Citation]):
        """Cache new citations"""
        try:
            citation_data = []
            for citation in citations:
                data = {
                    "citation_id": citation.citation_id,
                    "title": citation.title,
                    "authors": citation.authors,
                    "journal": citation.journal,
                    "publication_date": citation.publication_date.isoformat(),
                    "abstract": citation.abstract,
                    "doi": citation.doi,
                    "url": citation.url,
                    "citation_type": citation.citation_type,
                    "relevance_score": citation.relevance_score,
                    "keywords_matched": citation.keywords_matched,
                    "citation_context": citation.citation_context
                }
                citation_data.append(data)
            
            # Cache today's citations
            date_key = f"citations_{datetime.now().strftime('%Y%m%d')}"
            await self.cache.set(date_key, citation_data, ttl=86400 * 30)
            
            # Update running total
            existing_total = await self.cache.get("total_citations_count") or 0
            await self.cache.set("total_citations_count", existing_total + len(citations), ttl=86400 * 30)
            
        except Exception as e:
            self.logger.error(f"Error caching citations: {str(e)}")
    
    async def get_citation_metrics(self, days: int = 30) -> CitationMetrics:
        """Get citation metrics for specified period"""
        try:
            all_citations = []
            
            # Aggregate citations for the period
            for i in range(days):
                date = datetime.now() - timedelta(days=i)
                date_key = f"citations_{date.strftime('%Y%m%d')}"
                
                cached_citations = await self.cache.get(date_key)
                if cached_citations:
                    all_citations.extend(cached_citations)
            
            # Calculate metrics
            total_citations = len(all_citations)
            direct_citations = len([c for c in all_citations if c.get("citation_type") == "direct"])
            indirect_citations = len([c for c in all_citations if c.get("citation_type") == "indirect"])
            methodology_refs = len([c for c in all_citations if c.get("citation_type") == "methodology"])
            data_usage = len([c for c in all_citations if c.get("citation_type") == "data"])
            
            # Top citing journals
            journal_counts = {}
            for citation in all_citations:
                journal = citation.get("journal", "Unknown")
                journal_counts[journal] = journal_counts.get(journal, 0) + 1
            
            top_journals = sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_journals = [journal[0] for journal in top_journals]
            
            # Calculate growth rate (simplified)
            recent_count = len([c for c in all_citations if 
                              datetime.fromisoformat(c.get("publication_date", "")).date() > 
                              (datetime.now() - timedelta(days=7)).date()])
            growth_rate = recent_count / max(total_citations - recent_count, 1) * 100
            
            # H-index estimate (simplified)
            citation_counts = [c.get("relevance_score", 0) * 10 for c in all_citations]  # Proxy metric
            citation_counts.sort(reverse=True)
            h_index = 0
            for i, count in enumerate(citation_counts, 1):
                if count >= i:
                    h_index = i
                else:
                    break
            
            # Recent citations
            recent_citations = []
            for citation_data in all_citations[-5:]:  # Last 5
                citation = Citation(
                    citation_id=citation_data.get("citation_id", ""),
                    title=citation_data.get("title", ""),
                    authors=citation_data.get("authors", []),
                    journal=citation_data.get("journal", ""),
                    publication_date=datetime.fromisoformat(citation_data.get("publication_date", datetime.now().isoformat())),
                    abstract=citation_data.get("abstract", ""),
                    doi=citation_data.get("doi"),
                    url=citation_data.get("url", ""),
                    citation_type=citation_data.get("citation_type", ""),
                    relevance_score=citation_data.get("relevance_score", 0),
                    keywords_matched=citation_data.get("keywords_matched", []),
                    citation_context=citation_data.get("citation_context", "")
                )
                recent_citations.append(citation)
            
            return CitationMetrics(
                total_citations=total_citations,
                direct_citations=direct_citations,
                indirect_citations=indirect_citations,
                methodology_references=methodology_refs,
                data_usage_citations=data_usage,
                top_citing_journals=top_journals,
                citation_growth_rate=growth_rate,
                h_index_estimate=h_index,
                recent_citations=recent_citations
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating citation metrics: {str(e)}")
            return CitationMetrics(
                total_citations=0,
                direct_citations=0,
                indirect_citations=0,
                methodology_references=0,
                data_usage_citations=0,
                top_citing_journals=[],
                citation_growth_rate=0.0,
                h_index_estimate=0,
                recent_citations=[]
            )