from app.providers.crossref_provider import CrossrefProvider

def test_crossref_parses_metadata_and_sanitizes_abstract():
    data={"message":{"items":[{"title":["Paper"],"DOI":"10.123/a", "abstract":"<jats:p>A <b>result</b>.</jats:p>","published":{"date-parts":[[2024,1]]},"author":[{"given":"A","family":"B"}],"container-title":["Journal"]}]}}
    paper=CrossrefProvider.parse(data)[0]
    assert paper.abstract=="A result."
    assert paper.year==2024 and paper.authors==["A B"] and paper.url=="https://doi.org/10.123/a"
