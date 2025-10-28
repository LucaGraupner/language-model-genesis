Project Genesis: Training a 150M Parameter Language Model from Scratch
This repository documents my journey of training a medium-sized language model (approx. 150 million parameters) entirely from scratch. In an era where AI models are often used as magical black boxes via an API, I wanted to dig deeper. My goal was to personally navigate the entire lifecycle of a model: from selecting the infrastructure and preparing the data to managing the long-running training process and solving the countless real-world problems that arise.

The result is a base model that, after approx. 72 hours of training on an NVIDIA RTX 3090 using the ~5 GB wikitext corpus, is capable of generating coherent, grammatically complex sentences.

Tech Stack
Language & Framework: Python, PyTorch

Architectural Foundation: Andrej Karpathy's nanoGPT as a clean implementation of the Transformer architecture.

Cloud Infrastructure: On-demand GPU rental via the Vast.ai marketplace.

Development Environment: Remote connection via SSH, tmux for persistent sessions, scp and rsync for data transfer.

Tokenizer: tiktoken (OpenAI's GPT-2 tokenizer).

The Gauntlet: A Detailed Log of Challenges & Solutions
The path from idea to a final model was a true gauntlet. A significant part of the learning process was not writing code, but systematically debugging issues at every level of the stack. Here is an honest breakdown of the biggest hurdles and how I overcame them.

Phase 1: The Infrastructure Odyssey
1. Problem: Expired Free Credits
Challenge: The initial plan was to start the project on the Google Cloud Platform. However, I discovered that my free trial credits from a previous project had already expired.

Solution: A strategic provider pivot. Instead of abandoning the project, I evaluated alternatives and chose Vast.ai, a marketplace whose pay-as-you-go structure and low entry cost were ideal for this budget-conscious experiment.

2. Problem: Unreliable Hosts ("Retries exceeded")
Challenge: My first attempts to rent the absolute cheapest servers (e.g., in Japan for $0.101/hr) repeatedly failed with the status Error: retries exceeded.

Diagnosis: These were likely private machines that were offline or unstable. You cannot "reserve" a server and wait for it to come online.

Solution: Adapting my strategy. Instead of blindly chasing the lowest price, I learned to also consider the Reliability score and the Verified status. It is more efficient to pay a few cents more per hour for a stable server than to waste time on failed launch attempts.

3. Problem: The "Great Firewall of China"
Challenge: A server in China launched successfully, but the dataset download from Hugging Face failed with a ConnectionError.

Diagnosis: A simple ping huggingface.co in the SSH terminal confirmed that no packets were returning. The connection to the outside world was being blocked.

Solution: Recognizing the location as a critical factor and destroying the instance immediately. I learned that geographical location and potential network restrictions are crucial for data-intensive projects. Switching to a server in the US resolved the issue instantly.

Phase 2: Setup & Dependency Hell
4. Problem: Missing System Packages
Challenge: The attempt to start a persistent session with screen failed with command not found.

Diagnosis: The rented machine had a minimal OS that lacked this standard tool.

Solution: Manually installing the package. An apt-get update && apt-get install -y screen solved the problem. Later, it turned out that tmux was already pre-installed, which taught me to be flexible with different Linux environments.

5. Problem: Wrong Working Directory
Challenge: The first training attempt with python src/train.py failed with No such file or directory.

Diagnosis: I was executing the command from the home directory (/workspace or /root) instead of the project directory.

Solution: The simplest fix of all: cd language-model-genesis to move into the correct folder. A basic but essential lesson.

6. Problem: Missing Code Files
Challenge: The training wouldn't start, throwing a FileNotFoundError: 'configurator.py'.

Diagnosis: In my initial setup, I had only copied the core scripts (train.py, model.py) from nanoGPT, overlooking essential helper files.

Solution: Re-cloning the original repository, identifying the missing file, and using cp nanoGPT/configurator.py . to move it to the correct location. This highlighted the importance of complete and clean project setups.

7. Problem: Incorrect Data Structure
Challenge: After the configuration worked, the script failed again: FileNotFoundError: data/openwebtext/train.bin.

Diagnosis: My prepare.py script was placing the .bin files directly in the data/ folder, but train.py expected them in a specific subfolder.

Solution: Manually creating the expected folder structure and moving the files: mkdir data/openwebtext && mv data/*.bin data/openwebtext/.

Phase 3: The Training & Download Process
8. Problem: Training Restarted from Scratch
Challenge: After pausing the training at 31 hours to test it, restarting it caused the process to begin again at step 0 with a loss of 11.0.

Diagnosis: The default python src/train.py command always starts a new training run. It doesn't automatically load a checkpoint.

Solution: Using the correct command to resume: python src/train.py --init_from=resume. This was a critical lesson in managing long-running training processes.

9. Problem: Unstable scp Connection
Challenge: When trying to download the final 1.4 GB ckpt.pt file with scp, the connection repeatedly dropped (Connection reset).

Diagnosis: scp cannot resume interrupted downloads. It is unreliable for large files over unstable connections.

Solution: Switching to a professional tool. I installed the Windows Subsystem for Linux (WSL) and used the rsync command. rsync was able to resume the transfer from the exact point of failure, reliably getting the file onto my laptop.

10. Problem: scp Path Errors
Challenge: Even with scp, I got the No such file or directory error.

Diagnosis: I first used ~ in the path, which scp does not expand. Then I assumed /root, but the file was actually in /workspace.

Solution: Connecting via SSH and using find / -name ckpt.pt to find the exact, absolute path of the file and then using that path in the scp command.

Phase 4: Local Execution
11. Problem: Local Dependencies & CUDA Errors
Challenge: The sample.py script failed locally because torch was missing. After installation, a RuntimeError: CUDA is not available occurred.

Diagnosis: My local laptop did not have the same software environment as the server. The model was saved for a GPU but was being run on a CPU.

Solution: First, installing the local dependencies (torch, tiktoken). Then, modifying the code in sample.py to force execution on the CPU: device = 'cpu' and map_location='cpu'.

How to Use the Model
The final model (ckpt.pt) is located in the out folder. To run it locally on a CPU:

Install Dependencies:

Bash

# PyTorch for CPU (or GPU if available)
pip install torch torchvision torchaudio

# The tokenizer
pip install tiktoken
Adapt the Code: In src/sample.py, modify the following lines to force CPU usage:

device = 'cpu'

checkpoint = torch.load(ckpt_pt, map_location='cpu')

Generate Text:

Bash

python src/sample.py
What I've Learned
This project was one of the most intense and educational experiences of my programming journey. The takeaways go far beyond just writing Python code.

Theory Meets Brutal Reality: It's one thing to understand the Transformer architecture from a paper. It's something else entirely to overcome the practical hurdles required to make it work. The theory is just the ticket to entry; the real engineering work begins after.

DevOps is 50% of the Work: I spent more time in the command line with SSH, tmux, rsync, network diagnostics, and debugging Linux environments than I did in the code editor. Managing cloud infrastructure is not a sideline; it is a core competency for any serious AI project.

Systematic Debugging is Everything: I was confronted with errors at every level of the stack—from the network and disk space to subtle software conflicts. The ability to systematically isolate a problem (ping, ls -l, find, manual installations) was the deciding factor for success. I learned to read error messages not as roadblocks, but as precise directions to the solution.

VRAM is King (but not everything): The trade-off between VRAM, TFLOPS, and price per hour is an art form. I learned firsthand why an RTX 3090 with 24 GB of VRAM was often more valuable for this project than a theoretically faster card with less VRAM or a slower card with more.

Ultimately, this project gave me a deep, fundamental understanding and immense respect for the engineering effort that goes into large-scale AI products. It is the difference between driving a car and building the engine from scratch.
