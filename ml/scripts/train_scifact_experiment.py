"""Small reproducible SciFact training experiment; not a production quality claim.

Run with a GPU for a representative experiment. Uses only train qrels for fitting
and validation, and reserves BEIR test qrels for the final comparison.
"""
from __future__ import annotations
import argparse, json, random, time
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from torch.utils.data import DataLoader
import torch

BASE='cross-encoder/ms-marco-MiniLM-L2-v2'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-queries', type=int, default=40)
    parser.add_argument('--val-queries', type=int, default=8)
    parser.add_argument('--test-queries', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    args=parser.parse_args()
    random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
    corpus=load_dataset('BeIR/scifact','corpus',split='corpus')
    queries=load_dataset('BeIR/scifact','queries',split='queries')
    qrels_train=load_dataset('BeIR/scifact-qrels',split='train')
    qrels_test=load_dataset('BeIR/scifact-qrels',split='test')
    docs={str(x['_id']): ((x['title'] or '')+' '+(x['text'] or '')).strip() for x in corpus}
    query_text={str(x['_id']): x['text'] for x in queries}
    doc_ids=list(docs); texts=[docs[x] for x in doc_ids]
    vectorizer=TfidfVectorizer(stop_words='english',max_features=50000,ngram_range=(1,2))
    matrix=vectorizer.fit_transform(texts)
    def make_rows(qrels, count):
        positives=defaultdict(set)
        for row in qrels:
            if row['score']>0: positives[str(row['query-id'])].add(str(row['corpus-id']))
        ids=sorted(q for q in positives if q in query_text)
        random.Random(42).shuffle(ids)
        rows=[]
        for q in ids[:count]:
            pos=[docs[x] for x in sorted(positives[q]) if x in docs]
            if not pos: continue
            ranks=(matrix @ vectorizer.transform([query_text[q]]).T).toarray().ravel().argsort()[::-1]
            negatives=[docs[doc_ids[i]] for i in ranks if doc_ids[i] not in positives[q]][:4]
            rows.append({'query':query_text[q], 'positive':pos, 'negative':negatives})
        return rows
    trainval=make_rows(qrels_train,args.train_queries+args.val_queries)
    train=trainval[:args.train_queries]; val=trainval[args.train_queries:]
    test=make_rows(qrels_test,args.test_queries)
    if min(len(train),len(val),len(test))==0: raise RuntimeError('Empty split')
    assert not ({x['query'] for x in train} & {x['query'] for x in val+test})
    examples=[InputExample(texts=[row['query'],doc],label=1.0) for row in train for doc in row['positive']]
    examples += [InputExample(texts=[row['query'],doc],label=0.0) for row in train for doc in row['negative']]
    model=CrossEncoder(BASE,device='cpu',max_length=192,num_labels=1)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    evaluator=CrossEncoderRerankingEvaluator(val,at_k=5,name='validation',batch_size=8)
    test_eval=CrossEncoderRerankingEvaluator(test,at_k=5,name='heldout',batch_size=8,write_csv=False)
    before=test_eval(model,output_path=None)
    print('BASELINE',before,flush=True)
    model.fit(DataLoader(examples,shuffle=True,batch_size=8), evaluator=evaluator, epochs=args.epochs,
              warmup_steps=max(1,len(examples)//8//10), output_path=str(output),
              optimizer_params={'lr':2e-5}, show_progress_bar=True)
    trained=CrossEncoder(str(output),device='cpu',max_length=192)
    after=test_eval(trained,output_path=None)
    report={'base_model':BASE,'seed':42,'train_queries':len(train),'validation_queries':len(val),
            'test_queries':len(test),'pairs':len(examples),'epochs':args.epochs,
            'test_base':before,'test_finetuned':after,'test_protocol':'TF-IDF hard negatives, 4 per query; BEIR SciFact test qrels',
            'warning':'Small experiment on a narrow scientific fact-checking task, not a production benchmark.'}
    (output/'experiment.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
