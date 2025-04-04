### How to run the code
'''sh
./script/execute.sh -r -i ./queries/query-train.xml -o ranked.csv -m model -d /CIRB010
./script/execute.sh -r -i ./queries/query-test.xml -o ranked.csv -m model -d /CIRB010
'''

format :
-score: bm25
-feedback: true
-top_k : 100
-expansion_n: 10
-duplicated: true
-query: concepts + title + question