import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Load your JSON files
files = ['scores/bm25_scores.json', 'hybrid_scores.json', 'sparse_scores.json']
results = []

for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        method = file.replace('_scores.json', '').upper()
        
        # Flatten the metrics
        metrics = {}
        for metric_group in data:
            metrics.update(metric_group)
        
        results.append({'method': method, **metrics})

# Create DataFrame
df = pd.DataFrame(results)
df = df.sort_values('NDCG@10', ascending=False)

# Create interactive HTML report
html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Retrieval Results Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric-card {{ margin: 20px 0; padding: 20px; background: #f5f5f5; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .best {{ font-weight: bold; color: green; }}
    </style>
</head>
<body>
    <h1>Retrieval Results Analysis</h1>
    
    <div class="metric-card">
        <h2>Results Table</h2>
        {df.to_html(classes='results-table', escape=False)}
    </div>
    
    <div id="charts"></div>
    
    <script>
        const data = {df.to_dict(orient='records')};
        
        // Create charts
        const ndcgTrace = {{
            x: data.map(d => d.method),
            y: data.map(d => d['NDCG@10']),
            type: 'bar',
            name: 'NDCG@10'
        }};
        
        const recallTrace = {{
            x: data.map(d => d.method),
            y: data.map(d => d['Recall@100']),
            type: 'bar',
            name: 'Recall@100'
        }};
        
        Plotly.newPlot('charts', [ndcgTrace, recallTrace], {{
            title: 'Performance Comparison',
            barmode: 'group'
        }});
    </script>
</body>
</html>
"""

# Save the report
with open('results_report.html', 'w') as f:
    f.write(html)

print("Report generated: results_report.html")