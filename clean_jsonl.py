#!/usr/bin/env python3

import json
import sys
from pathlib import Path

def clean_jsonl(input_file: str, output_file: str) -> None:
    """Remove specified fields from each JSON object in a JSONL file."""
    
    fields_to_remove = {
        'emoji_usage',
        'emoji_count',
        'emoji_frequency',
        'max_length',
        'sentence_structure',
        'vocabulary_usage',
        'line_breaks',
        'avg_line_breaks',
        'punctuation_usage',
        'bullet_styles',
        'topic_shifts',
        'flow',
        'pacing',
        'sentiment_arc',
        'engagement_ratio',
        'tier'
    }
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                # Parse JSON object
                try:
                    obj = json.loads(line.strip())
                    
                    # Remove specified fields
                    for field in fields_to_remove:
                        obj.pop(field, None)  # Use pop with None to avoid KeyError if field doesn't exist
                    
                    # Write cleaned object
                    outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}", file=sys.stderr)
                    continue
                
        print(f"Successfully cleaned {input_file} and saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python clean_jsonl.py input.jsonl output.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    clean_jsonl(input_file, output_file)

if __name__ == '__main__':
    main() 