import json
import re
import sys
import csv

def load_source_data(filepath):
    """
    Loads source data into a list and a dictionary indexed by ID.
    """
    data_list = []
    data_by_id = {}
    data_by_title_lower = {}

    print(f"Loading source data from {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                data_list.append(item)
                data_by_id[item['id']] = item
                
                # Handle None title securely
                title_val = item.get('title')
                if title_val:
                    t_lower = title_val.strip().lower()
                    if t_lower:
                        if t_lower not in data_by_title_lower:
                            data_by_title_lower[t_lower] = []
                        data_by_title_lower[t_lower].append(item)
                        
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)
        
    print(f"Loaded {len(data_list)} source records.")
    return data_list, data_by_id, data_by_title_lower

def solve_ranking_question(question, data_list):
    """
    Solves questions like "highest rated movie from 1980".
    Returns: (new_answer, new_doc_ids_list) or (None, None)
    """
    q_lower = question.lower()
    
    # [FIX] Skip questions with topical constraints we can't check
    # If the question says "about", "subject", or "plot", we can't just sort by year.
    if "about " in q_lower or "subject" in q_lower:
        return None, None

    # Regex for variations: "highest rated movie", "movie with the highest rating"
    match = re.search(r'(highest|lowest|most|least)\s+(\w+)\s+(movie|tv show|show).*(\d{4})', question, re.IGNORECASE)
    
    if not match:
        # Try alternate pattern: "TV show in 1999 with the highest popularity"
        match = re.search(r'(movie|tv show|show) in (\d{4}).*(highest|lowest|most|least)\s+(\w+)', question, re.IGNORECASE)
        if match:
            m_type, m_year, m_adj, m_metric = match.groups()
        else:
            return None, None
    else:
        m_adj, m_metric, m_type, m_year = match.groups()

    target_year = int(m_year)
    target_type = "Movie" if "movie" in m_type.lower() else "TV"
    
    # Map synonyms to keys
    metric_map = {
        'rated': 'rating', 'rating': 'rating',
        'popularity': 'popularity', 'popular': 'popularity',
        'votes': 'votes', 'voted': 'votes'
    }
    
    sort_key = None
    for k, v in metric_map.items():
        if k in m_metric.lower():
            sort_key = v
            break
    if not sort_key: return None, None 

    reverse_sort = True if m_adj.lower() in ['highest', 'most'] else False

    # Filter candidates
    candidates = [
        x for x in data_list 
        if x.get('year') == target_year and x.get('type') == target_type and x.get(sort_key) is not None
    ]

    if not candidates:
        return "N/A", []

    # Sort to find the winner
    candidates.sort(key=lambda x: x[sort_key], reverse=reverse_sort)
    best_match = candidates[0]
    
    ans = best_match.get('title') or "Unknown Title"
    return ans, [best_match['id']]

def solve_attribute_question(question, current_gold_ids, data_by_id, data_by_title_lower):
    """
    Solves questions like "What is the rating of 'Bluefin'?"
    Prioritizes ID lookup to handle drift correctly.
    """
    q_lower = question.lower()

    # [FIX] Explicitly ignore "how long" questions (prevents matching "released" in description)
    if "how long" in q_lower:
        return None, None

    # Extract Title from quotes
    title_match = re.search(r'["\u201c](.+)["\u201d]', question)
    target_title = title_match.group(1) if title_match else None

    # Identify Question Target
    target_field = None
    
    if 'rating' in q_lower: target_field = 'rating'
    elif 'votes' in q_lower: target_field = 'votes'
    elif 'year' in q_lower: target_field = 'year'
    # [FIX] Stricter trigger for 'released'
    elif 'released' in q_lower:
        # Only map to year if the user is asking WHEN it was released
        if 'when' in q_lower or 'what year' in q_lower:
            target_field = 'year'
        else:
            return None, None
    
    if not target_field:
        return None, None

    # Strategy 1: Trust the ID
    doc_match = None
    if current_gold_ids:
        primary_id = current_gold_ids[0]
        if primary_id in data_by_id:
            doc_match = data_by_id[primary_id]

    # Strategy 2: Fallback to Title Search
    if not doc_match and target_title:
        candidates = data_by_title_lower.get(target_title.lower(), [])
        if candidates:
            doc_match = candidates[0]

    if doc_match:
        val = doc_match.get(target_field, "N/A")
        return str(val), [doc_match['id']]
    
    return None, None

def process_gold_set(source_path, gold_input_path, gold_output_path, report_path):
    source_list, source_map, source_title_map = load_source_data(source_path)

    updates_count = 0
    drift_changes = []

    print(f"Processing questions from {gold_input_path}...")
    
    try:
        with open(gold_input_path, 'r', encoding='utf-8') as fin, \
             open(gold_output_path, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                if not line.strip(): continue
                q_obj = json.loads(line)
                
                q_text = q_obj.get('question', '')
                q_id = q_obj.get('question_id', 'Unknown')
                original_ans = q_obj.get('gold_answer')
                original_ids = q_obj.get('gold_doc_ids', [])
                
                new_ans = None
                new_ids = None
                
                # --- Router Logic ---
                # 1. Superlative / Ranking
                if any(x in q_text.lower() for x in ['highest', 'most ', 'lowest', 'least ']):
                    new_ans, new_ids = solve_ranking_question(q_text, source_list)
                
                # 2. Attribute Lookup
                # Note: We check quotes to ensure it's targeting a specific entity
                elif any(x in q_text.lower() for x in ['rating', 'votes', 'year', 'released']) and ('"' in q_text or '\u201c' in q_text):
                    new_ans, new_ids = solve_attribute_question(q_text, original_ids, source_map, source_title_map)

                # 3. Static/Descriptive (or Fallback from above)
                # If functions return None, they fall through to here
                
                if new_ans is None:
                    # Keep original answer logic
                    valid_ids = [uid for uid in original_ids if uid in source_map]
                    if valid_ids:
                        new_ans = original_ans
                        new_ids = valid_ids
                    else:
                        new_ans = original_ans
                        new_ids = original_ids

                # --- Update & Track Drift ---
                if new_ans is not None:
                    if str(new_ans) != str(original_ans):
                        drift_changes.append({
                            'question_id': q_id,
                            'question': q_text,
                            'old_answer': original_ans,
                            'new_answer': new_ans
                        })
                    
                    q_obj['gold_answer'] = str(new_ans)
                    q_obj['gold_doc_ids'] = new_ids
                    updates_count += 1
                
                fout.write(json.dumps(q_obj) + '\n')
                
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Write CSV Report
    if drift_changes:
        print(f"Writing drift report with {len(drift_changes)} changes to {report_path}...")
        try:
            with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['question_id', 'question', 'old_answer', 'new_answer']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(drift_changes)
        except Exception as e:
             print(f"Error writing CSV report: {e}")
    else:
        print("No drift detected. No CSV report generated.")

    print(f"Done. Processed {updates_count} questions.")
    print(f"New gold set written to: {gold_output_path}")

if __name__ == "__main__":
    SOURCE_FILE = 'source_data.ndjson'
    GOLD_INPUT_TEMPLATE = 'gold_set_template.jsonl'
    GOLD_OUTPUT_FINAL = 'gold_set.jsonl'
    DRIFT_REPORT = 'drift_report.csv'
    
    process_gold_set(SOURCE_FILE, GOLD_INPUT_TEMPLATE, GOLD_OUTPUT_FINAL, DRIFT_REPORT)