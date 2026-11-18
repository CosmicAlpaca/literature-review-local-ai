"""Crossref metadata search without a provider API key."""
from html.parser import HTMLParser
import httpx
from app.providers.base import PaperResult

class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.parts=[]
    def handle_data(self, data): self.parts.append(data)
    def handle_endtag(self, tag):
        if tag.lower() in {"p", "jats:p", "br", "div"}: self.parts.append(" ")

def _plain(value):
    if not isinstance(value,str): return None
    parser=_Text(); parser.feed(value)
    return ' '.join(''.join(parser.parts).split()) or None

class CrossrefProvider:
    def fetch_papers(self, query: str, limit: int = 20) -> list[PaperResult]:
        if not query.strip() or not 1 <= limit <= 100: raise ValueError('Invalid query or limit')
        with httpx.Client(timeout=20,headers={'User-Agent':'LiteratureReviewAI/1.0'}) as client:
            response=client.get('https://api.crossref.org/works',params={'query.bibliographic':query,'rows':limit,'filter':'type:journal-article'})
            response.raise_for_status(); payload=response.json()
        return self.parse(payload)

    @staticmethod
    def parse(payload):
        papers=[]
        for item in payload.get('message',{}).get('items',[]):
            title=_plain((item.get('title') or [''])[0]); doi=item.get('DOI')
            if not title or not isinstance(doi,str): continue
            dates=(item.get('published') or item.get('issued') or {}).get('date-parts') or []
            year=dates[0][0] if dates and dates[0] and isinstance(dates[0][0],int) else None
            authors=[' '.join(filter(None,[a.get('given'),a.get('family')])) for a in item.get('author',[]) if isinstance(a,dict)]
            source=(item.get('container-title') or [None])[0]
            papers.append(PaperResult(title=title,abstract=_plain(item.get('abstract')),year=year,authors=authors,
                doi=doi,url='https://doi.org/'+doi,source_title=source,citation_count=item.get('is-referenced-by-count'),
                provider='crossref',raw_metadata={'crossref_type':item.get('type')}))
        return papers

    def search(self, query: str, page: int = 1): return self.fetch_papers(query)
