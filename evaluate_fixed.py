import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import gc

# Load model and tokenizer
print("Loading MedCPT model...")
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 第一块 GPU
    torch.cuda.set_device(0)         # 可选：显式设置 GPU
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
model = model.to(device)  # Move model to GPU if available
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def get_medcpt_embeddings_batch(queries, batch_size=32):
    """
    Process queries in batches to avoid memory issues
    """
    from tqdm import tqdm
    all_embeddings = []
    
    with torch.no_grad():
        # Create progress bar
        total_batches = (len(queries) - 1) // batch_size + 1
        pbar = tqdm(range(0, len(queries), batch_size), 
                   desc="Processing batches", 
                   total=total_batches,
                   unit="batch")
        
        for i in pbar:
            batch_queries = queries[i:i+batch_size]
            batch_num = i // batch_size + 1
            pbar.set_description(f"Processing batch {batch_num}/{total_batches}")
            
            encoded = tokenizer(
                batch_queries, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=64,
            )
            
            # Move input tensors to the same device as model
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            all_embeddings.append(embeds.cpu())  # Move to CPU to save GPU memory
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
    
    # Concatenate all embeddings
    final_embeddings = torch.cat(all_embeddings, dim=0)
    return final_embeddings, final_embeddings.size()

# Load data
print("Loading background data...")
back = pd.read_csv("background.csv", header=0, index_col=0)
back["Index"] = back.index

all_Ref = []
for term in back["Term"]:
    all_Ref.append(term)
print(f"Total background terms: {len(all_Ref)}")

# Load agent and GPT terms
print("Loading agent and GPT terms...")
def process_text(text: str) -> list:
    import re
    pattern = r'\([^)]*\)'
    segments = text.split('//')
    # Remove numbers and stop tokens ('-', '*')
    cleaned_segments = []
    for segment in segments:
        cleaned_segment = ''.join(char for char in segment)
        cleaned_segment = re.sub(pattern, '', cleaned_segment)
        cleaned_segment = cleaned_segment.replace('/', ' ').replace(","," ").replace("\"","").replace("-", " ").strip()
        if cleaned_segment:
            cleaned_segments.append(cleaned_segment)

    return cleaned_segments

# Read results of GeneAgent
agent = ""
with open ("Outputs/GeneAgent/Cascade/MsigDB_Final_Response_GeneAgent.txt", "r") as agentfile:
    for line in agentfile.readlines():
        agent += line
agent_text = process_text(agent)
agent_term = []
for text in agent_text:
    seg = text.split("\n")
    if len(seg) > 1:
        agent_term.append(seg[0].split(": ")[1])
    else:
        agent_term.append("None")
        
# Read results of GPT4
gpt = ""
with open ("Outputs/GPT-4/MsigDB_Response_GPT4.txt", "r") as gptfile:
    for line in gptfile.readlines():
        gpt += line
gpt_text = process_text(gpt)
gpt_term = []
for text in gpt_text:
    seg = text.split("\n")
    if len(seg) > 1:
        gpt_term.append(seg[0].split(": ")[1])
    else:
        gpt_term.append("None")

print(f"Agent terms: {len(agent_term)}")
print(f"GPT terms: {len(gpt_term)}")

# Process embeddings in batches
print("Processing reference embeddings...")
ref_embeds, ref_embeds_size = get_medcpt_embeddings_batch(all_Ref, batch_size=16)  # Smaller batch size for large dataset

print("Processing agent embeddings...")
agent_embeds, agent_embeds_size = get_medcpt_embeddings_batch(agent_term, batch_size=32)

print("Processing GPT embeddings...")
gpt_embeds, gpt_embeds_size = get_medcpt_embeddings_batch(gpt_term, batch_size=32)

print(f"Reference embeddings shape: {ref_embeds_size}")
print(f"Agent embeddings shape: {agent_embeds_size}")
print(f"GPT embeddings shape: {gpt_embeds_size}")

# Load reference data for comparison
print("Loading reference data...")
data = pd.read_csv("Datasets/MsigDB/MsigDB.csv", header=0, index_col=None)
reference = []
for term in data["Name"]:
    term = term.replace('/', ' ').replace(","," ").replace("\"","").replace("-", " ").strip()
    reference.append(term)

# Calculate similarities
print("Calculating similarities...")
agent_scores = []
gpt_scores = []

# Process reference embeddings for the specific dataset
print("Processing reference embeddings for comparison...")
ref_comparison_embeds, _ = get_medcpt_embeddings_batch(reference, batch_size=32)

from tqdm import tqdm

for i, (ref_embed, agent_embed, gpt_embed) in enumerate(tqdm(zip(ref_comparison_embeds, agent_embeds, gpt_embeds), 
                                                              desc="Calculating similarities", 
                                                              total=len(reference))):
    with torch.no_grad():
        score_agent = cos_sim(ref_embed.unsqueeze(0), agent_embed.unsqueeze(0))
        agent_scores.append(score_agent.item())
        
        score_gpt = cos_sim(ref_embed.unsqueeze(0), gpt_embed.unsqueeze(0))
        gpt_scores.append(score_gpt.item())

print(f"Average agent score: {np.average(agent_scores)}")
print(f"Average GPT score: {np.average(gpt_scores)}")
print(f"Max agent score: {np.max(agent_scores)}")
print(f"Max GPT score: {np.max(gpt_scores)}")

# ============================================================================
# 1-2. Calculate ROUGE scores
# ============================================================================

print("\n1-2. Calculating ROUGE scores...")
from rouge_score import rouge_scorer

if __name__ == "__main__":
    metrics = ["rouge1", "rouge2", "rougeL"]
    metric2results = {metric: [] for metric in metrics}
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
 
    for ref, hypagent in zip(reference, agent_term):
        scores_agent = scorer.score(ref, hypagent)
        for metric in metrics:
            metric2results[metric].append(scores_agent[metric].fmeasure)
   
    f = open("MsigDB.Rouge.txt","a")
    f.write("\n====GeneAgent (Cascade)====\n")
    for metric in metrics:
        results = metric2results[metric]
        rouge_score = sum(results) / len(results)
        f.write(metric + ":" + str(rouge_score) + "\n")
        print(f"{metric}: {rouge_score:.4f}")
    f.close()

# ============================================================================
# 2-1. Collect background gene sets
# ============================================================================

print("\n2-1. Collecting background gene sets...")
import csv

GSS = []
index = 0

# Add BP terms
try:
    bp = pd.read_csv("BP_terms_All.csv", header=0, index_col=0)
    for ID, Genes, Count, Name in zip(bp["GO"], bp["Genes"], bp["Gene_Count"], bp["Truth Label"]):
        GSS.append([index, ID, Genes, Count, Name])
        index += 1
    print(f"Added {len(bp)} BP terms")
except FileNotFoundError:
    print("Warning: BP_terms_All.csv not found, skipping BP terms")

# Add NeST terms
try:
    with open("Datasets/NeST/NeST.tsv", "r") as nestfile:
        for line in nestfile.readlines()[1:]:
            arr = line.split("\t")
            ID = arr[0]        # NEST ID
            Name = arr[1]      # name_new  
            Genes = arr[2].replace('"', '').replace(',', ' ').strip()  # Genes
            Count = len(Genes.split())
            GSS.append([index, ID, Genes, Count, Name])
            index += 1
    print(f"Added NeST terms, total: {len(GSS)}")
except FileNotFoundError:
    print("Warning: NeST.tsv not found, skipping NeST terms")

# Add MsigDB terms
zen = pd.read_csv("Datasets/MsigDB/MsigDB.csv", header=0, index_col=None)
for ID, Genes, Count, Name in zip(zen["ID"], zen["Genes"], zen["Count"], zen["Name"]):
    GSS.append([index, ID, Genes, Count, Name])
    index += 1

print(f"Final background gene sets count: {len(GSS)}")

# Save background gene sets
with open("background.csv", mode='w', newline='\n', encoding='utf-8') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)  
    writer.writerow(['Index', 'ID', 'Genes', 'Count', 'Term'])
    for term in GSS:
        writer.writerow(term)

print("Background gene sets saved to background.csv")

# Save results
# Calculate relative similarities between agent/GPT embeddings and background embeddings
print("Calculating relative similarities...")
agent_relative = cos_sim(agent_embeds, ref_embeds)
gpt_relative = cos_sim(gpt_embeds, ref_embeds)

print("Agent relative similarities shape:", agent_relative.shape)
print("GPT relative similarities shape:", gpt_relative.shape)

# Calculate ranks for agent results
print("Calculating agent ranks...")
AllDATA = pd.DataFrame({'Index': range(len(agent_term))})
relative = agent_relative.cpu().numpy()

# 修复索引错误的代码
scores = np.asarray(relative)
print(f"Scores shape: {scores.shape}")
print(f"AllDATA Index range: {AllDATA['Index'].min()} - {AllDATA['Index'].max()}")

rank = []
for row, ind in enumerate(tqdm(AllDATA["Index"], desc="Calculating ranks")):
    # 确保索引在有效范围内
    if row >= scores.shape[0]:
        print(f"Warning: row {row} exceeds scores array bounds")
        break
    if ind >= scores.shape[1]:
        print(f"Warning: index {ind} exceeds scores array bounds")
        continue
        
    root = scores[row][ind]
    ith = 1
    for j in range(scores.shape[1]):
        if scores[row][j] > root:
            ith += 1 
    rank.append(ith)

print(f"Calculated {len(rank)} ranks")
np.savetxt("MsigDB.Relative.Rank.GPT.Background.txt", np.asarray(rank), fmt="%s", newline="\n")

print("Saving results...")
np.savetxt("MsigDB.GeneAgent.Cascade.Semantic.csv", np.asarray(agent_scores), fmt="%s", delimiter="\t", newline="\n") 
np.savetxt("MsigDB.GPT4.Semantic.csv", np.asarray(gpt_scores), fmt="%s", delimiter="\t", newline="\n")

# ============================================================================
# 3. Evaluation for multiple enrichment terms test
# ============================================================================

print("\n" + "="*60)
print("SECTION 3: Multiple Enrichment Terms Test")
print("="*60)

# 3-1. Process output of GPT-4 in summarizing multiple enrichment terms
print("3-1. Processing GPT-4 summarized enrichment terms...")

try:
    text = ""
    with open("Outputs/EnrichedTermTest/gpt.geneagent.msigdb.summary.result.verification.txt", "r") as summary:
        for line in summary.readlines():
            text += line
    
    segments = text.split('//')
    print(f"Found {len(segments)} segments")
    
    enrich_terms = []
    for segment in segments:
        cleaned_segment = ''.join(char for char in segment)
        enrich = cleaned_segment.split("\n\n")[-2].replace(".", "").replace("\n","")
        enrich_terms.append(enrich.split("Enriched Terms: ")[1].split("; "))
        
    print(f"Processed {len(enrich_terms)} enrichment term sets")
    
    # 3-2. Exact match with all significant enrichment terms
    print("3-2. Matching with all significant enrichment terms...")
    
    import json
    with open("GSEATerms/MsigDB.EnrichTerms.Allsignificant.json","r") as file:
        enrich = json.load(file)
        
    name2id = {}
    for names in enrich:
        for name in names:
            if name["name"].lower() in name2id.keys():
                name2id[name["name"].lower()].append(name["native"])
            else:
                name2id[name["name"].lower()] = [name["native"]]

    results = []
    for terms in enrich_terms:
        matched = {} 
        for term in terms:
            if term.lower() in name2id.keys():
                matched[term] = list(set(name2id[term.lower()]))
            else:
                matched[term] = "None"
        results.append(matched)
            
    with open("Term2Enrich_Exact.Verification.Allsignificant.json", "w") as file:
        json.dump(results, file, indent=4)
    
    # Calculate match statistics
    with open("Term2Enrich_Exact.Verification.Allsignificant.json","r") as enrichfile:
        data = json.load(enrichfile)
    
    total, success, fail = 0, 0, 0
    for terms in data:
        for key in terms.keys():
            total += 1
            if terms[key] != "None":
                success += 1
            else:
                fail += 1

    print(f"Total summarized terms: {total}")
    print(f"Successful matches: {success}")
    print(f"Failed matches: {fail}")
    print(f"Match rate: {float(success/total):.4f}")
    
    # 3-3. Exact match with top-k (k=1,3,5) significant enrichment terms
    print("3-3. Matching with top-k enrichment terms...")
    
    # Process terms for matching
    for ind in range(0, len(enrich_terms)):
        for pos in range(0, len(enrich_terms[ind])):
            enrich_terms[ind][pos] = enrich_terms[ind][pos].lower().replace("-"," ")
    
    # TOP 1 matching
    with open("GSEATerms/MsigDB.EnrichTerms.top1.json","r") as file:
        enrich_top1 = json.load(file)

    pairs = []
    for terms, data in zip(enrich_terms, enrich_top1):
        temp = {}
        if data["name"].lower().replace("-"," ") in terms:
            temp[data["name"]] = data["native"]
        else:
            temp[data["name"]] = "None"
        pairs.append(temp)
        
    with open("Term2Enrich_Exact.Verification.top1.json", "w") as file:
        json.dump(pairs, file, indent=4)
    
    # TOP 5 matching
    with open("GSEATerms/MsigDB.EnrichTerms.top5.json","r") as file:
        enrich_top5 = json.load(file)
        
    name2id_top5 = {}
    for names in enrich_top5:
        for name in names:
            if name["name"].lower() in name2id_top5.keys():
                name2id_top5[name["name"].lower()].append(name["native"])
            else:
                name2id_top5[name["name"].lower()] = [name["native"]]
        
    results_top5 = []
    for terms in enrich_terms:
        matched = {} 
        for term in terms:
            if term.lower() in name2id_top5.keys():
                matched[term] = list(set(name2id_top5[term.lower()]))
            else:
                matched[term] = "None"
        results_top5.append(matched)
            
    with open("Term2Enrich_Exact.Verification.Top5.json", "w") as file:
        json.dump(results_top5, file, indent=4)
    
    # Calculate TOP 5 match statistics
    total_top5, success_top5 = 0, 0
    for terms in results_top5:
        total_top5 += 1
        for key in terms.keys():
            if terms[key] != "None":
                success_top5 += 1
                break
    
    print(f"TOP-5 total terms: {total_top5}")
    print(f"TOP-5 successful matches: {success_top5}")
    print(f"TOP-5 match rate: {float(success_top5/total_top5):.4f}")
    
except FileNotFoundError as e:
    print(f"Warning: Could not find enrichment test files: {e}")
    print("Skipping section 3 - enrichment terms test")

# ============================================================================
# 4. Other BERT-based model for semantic similarity evaluation
# ============================================================================

print("\n" + "="*60)
print("SECTION 4: Other BERT-based Models Evaluation")
print("="*60)

# 4-1. Sentence BERT evaluation
print("4-1. Sentence BERT evaluation...")

try:
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    torch.cuda.set_device(dev)

    from text2vec import Similarity
    sim_model = Similarity()
    sbert_scores = []
    
    for ref, hyp in zip(reference, agent_term):
        score = sim_model.get_score(ref, hyp)
        sbert_scores.append(score)

    print(f"Sentence BERT - Average score: {np.average(sbert_scores):.4f}")
    print(f"Sentence BERT - Max score: {np.max(sbert_scores):.4f}")
    
    # Save Sentence BERT results
    np.savetxt("MsigDB.SentenceBERT.Semantic.csv", np.asarray(sbert_scores), fmt="%s", delimiter="\t", newline="\n")
    
except ImportError:
    print("Warning: text2vec not available. Skipping Sentence BERT evaluation.")

# 4-2. SapBERT evaluation  
print("4-2. SapBERT evaluation...")

try:
    from transformers import AutoTokenizer, AutoModel
    
    sap_tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
    sap_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device)
    
    sapbert_scores = []
    for ref, hyp in zip(reference, gpt_term):
        toks = sap_tokenizer.batch_encode_plus([ref, hyp], 
                                           padding="max_length", 
                                           max_length=30, 
                                           truncation=True,
                                           return_tensors="pt")
        toks_cuda = {}
        for k,v in toks.items():
            toks_cuda[k] = v.to(device)
        cls_rep = sap_model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
        score = cos_sim(cls_rep[0], cls_rep[1])
        sapbert_scores.append(score.item())
        
    print(f"SapBERT - Average score: {np.average(sapbert_scores):.4f}")
    print(f"SapBERT - Max score: {np.max(sapbert_scores):.4f}")
    
    # Save SapBERT results
    np.savetxt("MsigDB.SapBERT.Semantic.csv", np.asarray(sapbert_scores), fmt="%s", delimiter="\t", newline="\n")
    
except Exception as e:
    print(f"Warning: SapBERT evaluation failed: {e}")

print("Processing completed successfully!")
