import os
import re

def read_data_file(data_path, verbose=False):
    """Read dataset file with format: input===>label
    
    Example formats supported:
    - [P][A][S][O][L]: sentence ===> output
    - [S][O][A][P][L]: sentence ===> output
    - sentence ===> output
    """
    sents, labels = [], []
    
    try:
        with open(data_path, 'r', encoding='UTF-8') as fp:
            for line in fp:
                line = line.rstrip("\n").strip()
                if not line or '===>' not in line:
                    continue
                
                try:
                    # Split by ===> to get input and output parts
                    parts = line.split('===>')
                    if len(parts) != 2:
                        if verbose:
                            print(f"Skipping invalid line: {line}")
                        continue
                    
                    input_part = parts[0].strip()
                    output_part = parts[1].strip()
                    
                    # Remove various prefix formats: [P][A][S][O][L]:, [S][O][A][P][L]:, etc.
                    input_part = re.sub(r'^\[\w\]\[\w\]\[\w\]\[\w\]\[\w\]:\s*', '', input_part)
                    
                    # Skip if input or output is empty
                    if not input_part or not output_part:
                        if verbose:
                            print(f"Skipping empty input/output: {line}")
                        continue
                    
                    # Handle multiple labels separated by ;
                    for label in output_part.split(';'):
                        sents.append(input_part)
                        labels.append(label.strip())
                except Exception as e:
                    if verbose:
                        print(f"Error processing line: {line}. Error: {e}")
    except FileNotFoundError:
        print(f"File not found: {data_path}")
    except Exception as e:
        print(f"Error reading file: {data_path}. Error: {e}")
    
    return sents, labels