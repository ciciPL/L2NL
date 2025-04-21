#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, RobertaTokenizer, \
    RobertaForCausalLM  # Keep these imports
import os # Added for creating output directory

from experiment.unixcoder.ESALE.rkt_code.unixcoder import UniXcoder

# --- Configuration ---
MODEL_ID = "unixcoder-base" # Explicitly use the full model identifier
# INPUT_FILE = "../../../../dataset/racket/code_rkt.txt" # <-- ADJUST THIS PATH if needed
INPUT_FILE = "../rkt_result/rkt_2_python_40510.txt" # <-- ADJUST THIS PATH if needed
OUTPUT_DIR = "../rkt_result" # <-- ADJUST THIS PATH if needed
OUTPUT_FILENAME = "result_rk_python_NL_base.txt" # Choose a descriptive name
BATCH_SIZE = 4  # Adjust based on your GPU memory (e.g., 4, 8, 16, 32)
MAX_SUMMARY_LENGTH = 30 # Max *new* tokens for the summary
MAX_INPUT = 256
# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# --- Load Model and Tokenizer ---
print(f"Loading tokenizer for {MODEL_ID}...")
# trust_remote_code might not be strictly needed for unixcoder-base, but kept from original

print(f"Loading model {MODEL_ID}...")
# Load the model first, then move to device
model = UniXcoder("unixcoder-base")
model.to(device) # Move the model to the selected device
model.eval() # Set model to evaluation mode
print("Model and tokenizer loaded successfully.")
# --- Generation Function ---
def generate_summary_batch(code_list):
    """Generates summaries for a batch of code strings."""
    prompts = [
        # Using the same prompt structure as your original code
        # You might experiment with different prompts for potentially better results
        f"""# <mask0>
            {code}"""
        for code in code_list
    ]

    # Tokenize the batch of prompts
    inputs_id = model.tokenize(
        prompts,
        padding=True,
        mode="<encoder-decoder>",
        max_length=MAX_INPUT # Use model's default max length
    ) # Move tokenized inputs to the same device as the model
    inputs = torch.tensor(inputs_id).to(device)
    # Generate summaries
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model.generate(
            inputs, decoder_only=False, beam_size=3, max_length=256
        )
        predictions = model.decode(outputs)
        result = [x.replace("<mask0>", "").strip() for x in predictions[0]]
        print(result)

    # # Decode generated sequences
    # # The output includes the prompt, so we need to remove it.
    # decoded_summaries = []
    # for output, prompt in zip(outputs, prompts):
    #     # Decode the full sequence
    #     full_decoded_text = tokenizer.decode(output, skip_special_tokens=True)
    #     # Remove the prompt part to isolate the generated summary
    #     # Use rfind('Summary:') to handle cases where 'Summary:' might appear in code
    #     summary_start_index = full_decoded_text.rfind('Summary:')
    #     if summary_start_index != -1:
    #         summary = full_decoded_text[summary_start_index + len('Summary:'):].strip()
    #     else: # Fallback if 'Summary:' isn't found (shouldn't happen with this prompt)
    #          summary = full_decoded_text.replace(prompt, "").strip() # Less robust fallback
    #     decoded_summaries.append(summary)

    return result


# --- File Processing Function ---
def process_file(input_file, output_file, batch_size):
    """Reads code from input file, generates summaries in batches, and writes to output file."""
    print(f"Processing file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"Using batch size: {batch_size}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    line_count = 0
    processed_count = 0
    error_count = 0
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        batch = []
        for line in infile:
            line_count += 1
            if ':' in line:
                index, code = line.split(':', 1)
                # Skip if code is empty after stripping
                code = code.strip()
                if code:
                     batch.append((index.strip(), code))
                else:
                    print(f"⚠️ Warning: Skipping empty code for index {index.strip()} at line {line_count}")
                    outfile.write(f"{index.strip()}: [Error: Empty code provided]\n")


                # When batch is full, process it
                if len(batch) >= batch_size:
                    indices, codes = zip(*batch)
                    try:
                        # Generate summaries for the current batch
                        summaries = generate_summary_batch(codes)
                        # Write summaries to the output file
                        for idx, summary in zip(indices, summaries):
                            outfile.write(f"{idx}: {summary}\n")
                        processed_count += len(batch)
                        print(f"Processed batch ending with index {indices[-1]} ({processed_count}/{line_count} lines processed approx.)")
                    except Exception as e:
                        # Handle errors during batch generation
                        error_count += len(batch)
                        print(f"❌ Error processing batch ending with index {indices[-1]}: {str(e)}")
                        # Write error message for each item in the failed batch
                        for idx in indices:
                            outfile.write(f"{idx}: [Error: {str(e)}]\n")
                        # Optionally add more detailed error logging or traceback here
                        # traceback.print_exc()
                    finally:
                         # Clear the batch regardless of success or failure
                        batch = []

        # Process any remaining items in the last batch
        if batch:
            print(f"Processing final batch of size {len(batch)}...")
            indices, codes = zip(*batch)
            try:
                summaries = generate_summary_batch(codes)
                for idx, summary in zip(indices, summaries):
                    outfile.write(f"{idx}: {summary}\n")
                processed_count += len(batch)
                print(f"Processed final batch ending with index {indices[-1]}")
            except Exception as e:
                error_count += len(batch)
                print(f"❌ Error processing final batch ending with index {indices[-1]}: {str(e)}")
                for idx in indices:
                     outfile.write(f"{idx}: [Error: {str(e)}]\n")
                # traceback.print_exc()

    print("-" * 30)
    print(f"File processing complete.")
    print(f"Total lines read: {line_count}")
    print(f"Successfully processed items: {processed_count}")
    print(f"Items with errors: {error_count}")
    print(f"Results saved to: {output_file}")
    print("-" * 30)


# --- Main Execution ---
if __name__ == "__main__":
    # Construct full output path
    full_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    print("Starting code summarization using UnixCoder...")
    process_file(INPUT_FILE, full_output_path, BATCH_SIZE)
    print("Summarization finished.")