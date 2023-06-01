import glob

search_result_texts = glob.glob("AmazonBooks_new_default/*.txt")

max_ndcg_10 = -1
trace_id_ndcg_10 = 0

for search_result_text in search_result_texts:
    f = open(search_result_text, "r", encoding="utf8")
    logs = [log.replace("\n", "").replace("(", "").replace(")", "") for log in f.readlines()]
    for log in logs:
        NDCG_10 = float(" ".join(log.split(",")[0].split()))
        max_ndcg_10 = max(max_ndcg_10, NDCG_10)

for search_result_text in search_result_texts:
    f = open(search_result_text, "r", encoding="utf8")
    logs = [log.replace("\n", "").replace("(", "").replace(")", "") for log in f.readlines()]
    for log in logs:
        NDCG_10 = float(" ".join(log.split(",")[0].split()))
        if max_ndcg_10 == NDCG_10:
            print(search_result_text)
            exit()