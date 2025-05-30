# **Structuring Your AI/LLM Learning Journey on GitHub: A Practical Guide**

## **I. Foreword: Structuring Your Learning Journey on GitHub**

Embarking on a significant learning endeavor, particularly in a rapidly evolving field like Artificial Intelligence (AI) and Large Language Models (LLMs), benefits immensely from a structured approach to documentation and progress tracking. Creating a dedicated GitHub repository for this purpose offers several advantages. It reinforces learning through the act of articulation, as explaining concepts helps solidify understanding. Such a repository can become a valuable portfolio piece, showcasing commitment and organizational skills to potential collaborators or employers. Furthermore, by openly documenting the journey, it can serve as a guide and inspiration for others treading a similar path, fostering a sense of community and shared knowledge, which is central to the open-source ethos prevalent in the AI field.1

GitHub, traditionally known as a platform for code hosting and version control, is also an exceptionally powerful tool for structured documentation. Its support for Markdown allows for richly formatted notes, while features like Issues can be adapted for question-and-answer sections or to-do lists, and Project Boards can manage more complex learning milestones.

A crucial perspective to adopt when creating such a repository is to treat it as a **dynamic, living document**. The goal is not merely to create a static plan, but rather a chronicle that evolves with one's understanding and progress. This approach aligns perfectly with the user's intent to "track my own progress." A static plan outlines intent, whereas a living document actively reflects ongoing learning, challenges encountered, and achievements unlocked. GitHub's inherent version control capabilities are ideal for tracking the evolution of this learning log, capturing the journey itself. This dynamism makes the repository more valuable for personal reflection—allowing one to see how far they've come and identify tricky concepts—and more authentic and useful for others, who can observe a realistic learning process rather than a polished, after-the-fact summary. Therefore, frequent commits, status updates, and the addition of reflections as learning progresses are highly encouraged.

## **II. Choosing a Name and Description for Your Repository**

The name and description of the GitHub repository serve as its initial handshake with the world. They are the first elements a visitor (including the future self) encounters and should be crafted to be both informative and inviting, clearly stating the repository's value.

### **Repository Name Suggestions**

When selecting a name, several principles should guide the choice:

* **Memorability**: A name that is easy to recall will be more effective.  
* **Descriptiveness**: It should clearly indicate the learning domain. Given the context of the provided research materials, which heavily focus on AI and LLMs 3, names reflecting this domain are appropriate.  
* **Uniqueness**: While not strictly necessary, a unique name can help it stand out.  
* **Keywords**: Incorporating terms like "learning-journey," "deep-dive," "exploration," "study-log," or "path" can effectively communicate the repository's purpose.

Based on these principles, and assuming a learning path focused on AI/LLMs, here are some examples:

* LLM-Learning-Odyssey  
* AI-Deep-Dive-Path  
* My-LLM-Mastery-Journal  
* NLP-With-Transformers-Study (Suitable if the learning path is centered around a specific course like the Hugging Face NLP Course 6 or Stanford's CS224n 9)  
* Karpathy-Zero-To-Hero-Log (Appropriate if following a particular resource, such as Andrej Karpathy's "Neural Networks: Zero to Hero" series 11)  
* Applied-AI-Learning-Roadmap

### **Short Description Guidance**

The short description appears directly under the repository name on GitHub and provides a concise summary of its content and purpose, typically in one or two sentences.

* **Purpose**: To quickly inform visitors about what the repository contains.  
* **Keywords**: Including relevant keywords enhances discoverability. For an AI/LLM learning path, terms such as "LLM," "AI," "Deep Learning," "Machine Learning," "Learning Path," "Study Notes," "Progress Tracking," and names of key technologies or concepts (e.g., "Transformers," "Fine-tuning") are beneficial.  
* **Example (for LLM-Learning-Odyssey)**: "A personal journey documenting the study of Large Language Models (LLMs), from foundational concepts like Transformer architecture to advanced applications in fine-tuning and ethical AI. Includes notes, project summaries, and progress tracking."

The name and description act as a "hook." Since a goal is for the experience to be "useful for others too," a clear, engaging name and description function much like a compelling title and abstract for a research paper or book—they draw people in and set clear expectations. A vague name (e.g., my-study-notes) or an uninformative description will likely deter potential visitors. A well-crafted name and description not only enhance discoverability but also signal the creator's seriousness and organizational approach to their learning.

## **III. Crafting an Effective README.md**

The README.md file is the cornerstone of the GitHub repository. It serves as the main landing page and should be a comprehensive guide for both the creator and any visitors, outlining the learning path and the repository's structure. It should be treated as a dynamic dashboard for the learning journey, not merely a static table of contents.

### **Essential Sections for the README.md**

**A. Title and Introduction**

* **Title**: A clear, prominent title, such as "My Journey into Large Language Models" or "AI/LLM Learning Path & Progress."  
* **Introduction**:  
  * A brief personal motivation for undertaking this learning path.  
  * A concise overview of what the repository contains and what visitors can expect (e.g., "This repository documents my ongoing study of Large Language Models, including theoretical notes, code experiments, project summaries, and a detailed progress log.").

**B. Learning Goals and Scope**

* **Learning Goals**: Clearly articulate the specific objectives. For example: "To gain a comprehensive understanding of Transformer architectures 12, master techniques for pretraining 14 and fine-tuning LLMs (SFT, RLHF, PEFT) 15, explore the capabilities of models like Llama 3 19 and DeepSeek 21, and critically evaluate the ethical implications of AI.22"  
* **Scope**: Define the boundaries of the learning path. What topics are explicitly included? What related areas might be considered out of scope for now? This helps manage personal focus and visitor expectations.

C. Core Topics/Modules (The Learning Path Structure)  
This section forms the heart of the learning path documentation. It should be presented as a structured list or a series of sections, each representing a key learning area. The following modules are examples, inspired by the breadth of topics in the provided research material, and should be adapted to the user's specific learning plan.

* **Module 1: Foundations of AI and Machine Learning**  
  * Core Machine Learning Concepts: Supervised Learning (e.g., regression, classification), Unsupervised Learning (e.g., clustering).24  
  * Essential Mathematics: Linear Algebra (vectors, matrices, transformations) 25, Calculus (derivatives, chain rule, gradients for optimization) 25, Probability & Statistics (distributions, estimation).25  
  * Introduction to Deep Learning: Neural network basics, activation functions, loss functions, optimization.32  
* **Module 2: Python and Essential Libraries for AI/ML**  
  * Python Fundamentals: Core syntax, data structures, control flow, functions, OOP.34  
  * NumPy: Array manipulation, numerical operations.40  
  * PyTorch or TensorFlow: Building and training neural networks, tensor operations, autograd.42  
  * Jupyter Notebooks / Google Colab: Interactive coding, experimentation, visualization.46  
* **Module 3: Understanding Transformer Architecture**  
  * The Seminal Paper: "Attention Is All You Need".12  
  * Core Components: Encoder-decoder structure, Self-Attention mechanism (Query, Key, Value), Multi-Head Attention.13  
  * Positional Encoding: Addressing sequence order, including techniques like Rotary Positional Encoding (RoPE).60  
  * Visualizing Transformers: Understanding data flow and component interactions (e.g., using tools like bbycroft.net/llm 62 or Jay Alammar's "The Illustrated Transformer" 13).  
* **Module 4: LLM Pretraining**  
  * Data Lifecycle: Collection (e.g., Common Crawl 69), cleaning, filtering, deduplication. Understanding high-quality datasets like FineWeb and FineWeb-Edu.19 Challenges in pretraining data.80  
  * Tokenization Deep Dive: Byte-Pair Encoding (BPE) 82, SentencePiece 84, WordPiece. Understanding tokenizers like Tiktoken 86 and tools like tiktokenizer.vercel.app.86 The concept of "LLMs need tokens to think".95  
  * Model Initialization and Training Loop: Setting up model architecture, weight initialization, forward/backward pass, optimizers.14  
  * Scaling Laws: Relationship between model size, data size, and performance.100  
* **Module 5: LLM Fine-tuning Techniques**  
  * Supervised Fine-Tuning (SFT): Adapting pretrained models to specific tasks using labeled datasets.16 Role of conversational data.102  
  * Reinforcement Learning from Human Feedback (RLHF): Aligning models with human preferences using reward models and algorithms like Proximal Policy Optimization (PPO).15  
  * Parameter-Efficient Fine-Tuning (PEFT): Techniques like Low-Rank Adaptation (LoRA) for efficient fine-tuning.18  
  * Instruction Tuning: Training models to follow natural language instructions (e.g., InstructGPT 15).  
  * Model Merging: Combining multiple fine-tuned models.18  
* **Module 6: LLM Inference and Evaluation**  
  * Decoding Strategies: Autoregressive generation, greedy search, beam search, top-k sampling, top-p (nucleus) sampling.109  
  * Temperature Parameter: Controlling randomness and creativity in output.115  
  * Evaluation Metrics: Perplexity, BLEU, ROUGE, and task-specific metrics.116  
  * Benchmarking Platforms: Understanding leaderboards like LM Arena for model comparison.117  
* **Module 7: Exploring Key LLM Models and Architectures**  
  * Meta's Llama Series: Llama 3, Llama 3.1, Llama 3.2 – architecture (GQA, RoPE, SwiGLU), training data, performance.19  
  * DeepSeek Models: DeepSeek LLM, DeepSeek-Coder, DeepSeek-V2 (MoE, MLA), DeepSeek-R1 – architectures, data strategies, scaling laws, performance.130  
  * Other Notable Models: Surveying the landscape of open-source and proprietary LLMs.3  
* **Module 8: LLM Tools, Platforms, and Ecosystems**  
  * Hugging Face Ecosystem: Transformers library, Datasets library, Tokenizers library, Model Hub, Inference Playground, Spaces.156  
  * Local LLM Experimentation: Tools like LM Studio for running models on personal hardware.168  
  * Inference Platforms: Services like TogetherAI Playground 171 and Hyperbolic 174 for accessing and running various models.  
* **Module 9: Advanced LLM Concepts and Capabilities**  
  * Reasoning in LLMs: Techniques like Chain-of-Thought (CoT) prompting and the ReAct framework.177  
  * LLM-Powered Autonomous Agents: Planning, memory, and tool use in agentic systems.180  
  * LLM "Self-Knowledge" and "Theory of Mind": Exploring research on whether LLMs can model their own states or understand others' mental states.181  
  * "Jagged Intelligence" / "Jagged Frontier": Understanding the uneven capabilities of LLMs and benchmarks like SIMPLE.190  
* **Module 10: Ethical Considerations in AI/LLMs**  
  * Bias and Discrimination: Identifying and mitigating biases in training data and model outputs.22  
  * Misinformation and Malicious Use: Addressing the potential for LLMs to generate fake news or be used for harmful purposes.22  
  * Transparency and Explainability: The "black box" problem and efforts towards more interpretable AI.197  
  * Environmental Impact: The energy consumption and carbon footprint of training and running large models.22  
  * Accountability and Governance: Establishing responsibility for AI-driven decisions and actions.22  
  * Responsible AI Development Practices: Frameworks and guidelines for building AI systems ethically.23  
* **Module 11: Learning from Experts and Curated Educational Resources**  
  * Andrej Karpathy: "Neural Networks: Zero to Hero" series (Micrograd, Makemore, nanoGPT, minBPE) 11 and llm.c project.201  
  * DeepLearning.AI Specializations & Short Courses: Covering Prompt Engineering 203, LangChain 204, Fine-tuning 17, Pretraining 14, RLHF 107, Gradio for demos 163, LLMOps 206, Red Teaming 207, and more.  
  * University Courses: Stanford CS224n (NLP with Deep Learning) 9, fast.ai NLP course.209  
  * Seminal Papers and Key Research Areas: "Attention Is All You Need" 12, AlphaGo/AlphaZero research 212, ReAct framework.5

D. Key Resources  
This section should compile a curated list of direct links to essential learning materials.

* **Foundational Papers**:  
  * "Attention Is All You Need" (Vaswani et al., 2017\) 12  
  * InstructGPT Paper (Ouyang et al., 2022\) 15  
  * ReAct Paper (Yao et al., 2022\) 5  
  * AlphaGo/AlphaGo Zero Papers 212  
* **Model Documentation & Repositories**:  
  * Llama 3: Official Blog 121, Model Card 129, arXiv papers 19  
  * DeepSeek Models: Official Website 21, arXiv papers (DeepSeek LLM 100, DeepSeek-Coder 138, DeepSeek-V2 139, DeepSeek-R1 144)  
* **Courses & Tutorials**:  
  * Andrej Karpathy's "Neural Networks: Zero to Hero" & llm.c 11  
  * DeepLearning.AI Courses (links to specific courses like Pretraining 14, Fine-tuning 17, RLHF 107, etc.)  
  * Hugging Face Courses (NLP Course 6)  
  * Stanford CS224n 9  
* **Tools & Platforms**:  
  * Hugging Face:([https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)) 164,([https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)) 165,(https://huggingface.co/docs/tokenizers) 166, [Inference Playground](https://huggingface.co/inference-playground) 158  
  * LM Studio 169  
  * Tiktokenizer 86  
  * LLM Visualization (bbycroft.net/llm) 62

E. Progress Tracking Table  
This table is the core mechanism for tracking progress and making the repository useful. It transforms the README into an active dashboard, making it immediately clear to the user and others what is being worked on, what has been completed, and where detailed learnings are located. Regular updates to this table are crucial.

| Module/Topic | Status | My Notes/Code | Key Resources | Start Date | Completion Date | Confidence (1-5) | Key Learnings/Challenges |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Module 1: Foundations | ✅ Completed | [Notes](http://docs.google.com/modules/01-foundations/README.md) | [Coursera ML 24](http://docs.google.com/link), [Khan Academy Calculus 219](http://docs.google.com/link) | 2024-07-01 | 2024-07-15 | 4/5 | Linear algebra was a good refresher. |
| Module 3: Transformer Architecture | 🚧 In Progress | [Notes](http://docs.google.com/modules/03-transformer-architecture/) | [Attention Is All You Need 12](http://docs.google.com/link_to_paper),\](link\_to\_blog) | 2024-07-16 |  | 3/5 | Multi-head attention details are complex. |
| Module 4: LLM Pretraining | ⏳ To Do |  | [FineWeb Paper 70](http://docs.google.com/link_to_fineweb),\](link\_to\_course) |  |  |  | Data pipelines seem challenging. |
| Module 5: LLM Fine-tuning | ⏳ To Do |  | \](link\_to\_course),\](link) |  |  |  | Understanding PEFT methods. |
| Module 10: Ethical Considerations | ⏳ To Do |  | \](link),\](link) |  |  |  | Nuances of fairness and bias mitigation. |

*(Note: Links in the table above are placeholders and should be replaced with actual links by the user.)*

F. How to Use This Repository (For Others)  
A brief section explaining how others might navigate the repository, use the notes, or follow the learning path. For example: "This repository is structured by learning modules. Each module in the /modules/ directory contains a README.md with a summary and links to specific notes or code. The main progress table above provides an overview and direct links."  
G. (Optional) Contribution Guidelines  
If open to collaboration or suggestions: "Suggestions for resources or corrections are welcome\! Please open an Issue or submit a Pull Request." (Inspired by open-source contribution models 2).  
H. Tools and Environment Setup  
Briefly list primary tools or software used for learning (e.g., Python 3.10, PyTorch 2.1, VS Code, LM Studio). This is helpful for reproducibility if code projects are included.

### **Markdown Best Practices**

* Utilize headings (\#, \#\#, \#\#\#) for structure.  
* Employ bullet points (\* or \-) and numbered lists for clarity.  
* Use code blocks () for code snippets.  
* Embed links effectively for resources.  
* Keep the README updated as the central navigation point for the repository.

## **IV. Designing a Scalable Directory Structure**

A clean, well-organized directory structure is essential for managing learning materials effectively and making the repository navigable for both the creator and others. The structure should be modular, with clear, consistent naming, and scalable to accommodate new topics and projects. The directory structure should directly mirror the modular breakdown in the README.md for intuitive navigation.

### **Suggested Top-Level Directories**

* **/README.md**: The main landing page and guide (as detailed above).  
* **/modules/** (or /topics/): This directory will house subdirectories for each core learning module outlined in the README's progress table.  
  * Each module subdirectory (e.g., /modules/01-foundations/, /modules/03-transformer-architecture/) should ideally contain:  
    * README.md: A module-specific overview, key concepts, summaries of learning, links to primary resources for that module, and links to any associated projects.  
    * notes/: Markdown files with detailed personal notes, summaries of papers or videos, and reflections.  
    * code\_examples/: Any small, self-contained code snippets or scripts related to the module's concepts (e.g., a simple implementation of an attention head).  
    * papers/ (optional): If storing PDFs of key research papers, ensuring copyright and fair use are respected.  
* **/projects/**: This directory is for larger, more involved coding projects undertaken as part of the learning path.  
  * Each project should have its own subdirectory (e.g., /projects/minGPT-from-scratch/, /projects/document-summarizer-finetuning/).  
  * Project subdirectories should contain all relevant code, data (or scripts to download data), a project-specific README.md detailing the project's goals, methods, results, and challenges.  
* **/resources/**: A general collection of external resources that don't fit neatly into a specific module.  
  * cheatsheets/: Quick reference guides for libraries or concepts.  
  * glossary.md: Definitions of key terms encountered.  
  * tools\_setup.md: Notes on setting up specific software or environments.  
  * bibliography.bib or references.md: A centralized list of all cited papers and resources.  
* **/journal/** (or /logs/): For more informal, personal reflections, weekly progress updates, or a "lab notebook" style of tracking experiments, thoughts, and hurdles. This can be particularly useful for the "living document" philosophy.  
* **LICENSE**: A file specifying the licensing for the content (e.g., MIT for code, CC-BY-SA 4.0 for notes).  
* **.gitignore**: A standard file to instruct Git which files or directories to ignore (e.g., Python virtual environments (venv/, \_\_pycache\_\_/), IDE configuration files (.vscode/, .idea/), large datasets, compiled files).

### **Example Directory Structure**

.  
├── README.md  
├── LICENSE  
├──.gitignore  
├── modules/  
│   ├── 00-introduction-and-setup/  
│   │   └── README.md  
│   ├── 01-foundations/  
│   │   ├── README.md  
│   │   └── notes/  
│   │       ├── math\_for\_ml.md  
│   │       └── probability\_basics.md  
│   ├── 02-python-and-libraries/  
│   │   ├── README.md  
│   │   ├── notes/  
│   │   │   ├── numpy\_deep\_dive.md  
│   │   │   └── pytorch\_fundamentals.md  
│   │   └── code\_examples/  
│   │       └── simple\_neural\_network.py  
│   ├── 03-transformer-architecture/  
│   │   ├── README.md  
│   │   ├── notes/  
│   │   │   ├── attention\_mechanism\_explained.md  
│   │   │   └── positional\_encoding\_variants.md  
│   │   └── papers/  
│   │       └── vaswani\_et\_al\_2017\_attention.pdf  
│   ├──... (other modules as defined in README)...  
│   └── 11-learning-from-experts/  
│       ├── README.md  
│       └── notes/  
│           └── karpathy\_zero\_to\_hero\_summary.md  
├── projects/  
│   ├── project\_1\_custom\_tokenizer/  
│   │   ├── README.md  
│   │   ├── src/  
│   │   └── data/  
│   └── project\_2\_llm\_finetuning\_experiment/  
│       ├── README.md  
│       ├── scripts/  
│       └── results/  
├── resources/  
│   ├── cheatsheets/  
│   │   └── pytorch\_cheatsheet.pdf  
│   ├── glossary.md  
│   └── tools\_setup.md  
└── journal/  
    ├── week\_01\_reflections.md  
    └── llm\_paper\_thoughts.md

This structure ensures that as the learning journey progresses and materials accumulate, they remain organized and easily accessible. Each module in the main README's progress table should directly correspond to a directory within /modules/, creating a clear and consistent navigation path.

## **V. Content Ideas for Learning Modules**

The specific modules will depend on the individual's learning goals, but the research snippets provide a wealth of potential topics central to understanding AI and LLMs. The list of example modules provided in Section III.C of the README structure serves as a strong starting point. These modules cover a logical progression from foundational knowledge to advanced concepts and practical applications.

It is important to remember that a learning path is rarely strictly linear. Advanced topics often illuminate or necessitate a deeper understanding of fundamentals. For instance, debugging a complex Transformer model 13 might require revisiting the intricacies of gradient flow, as detailed in resources like Andrej Karpathy's "Backprop Ninja" segment of the "Neural Networks: Zero to Hero" course.11 Similarly, exploring the performance of different LLMs like Llama 3 127 or DeepSeek-V2 139 might prompt a deeper dive into their specific architectural innovations (e.g., GQA, RoPE, MLA, MoE) or the datasets they were trained on.70

The learning path should therefore be flexible, allowing for and encouraging this iterative deepening of knowledge. The progress tracking table in the README can accommodate this by using statuses like "🔄 Revisiting" or by adding notes in the "Key Learnings/Challenges" column that reflect how new concepts clarified older ones. Documenting these "aha\!" moments, when a new piece of information connects disparate concepts, is a valuable part of the learning process itself.

## **VI. Maintaining and Sharing Your Learning Journey**

### **Regular Updates**

Consistency is key to maintaining the value of this learning repository. Regular updates to notes, the progress table, and personal reflections ensure the repository remains a "living document." Establishing a routine, such as a weekly review and commit session, can help maintain momentum and keep the information current.

### **Using GitHub Features for Engagement**

If the user is open to interaction, GitHub's features can enhance the learning experience:

* **Issues**: Can be used for self-posed questions, tracking bugs in personal coding projects, or even as a space for others to ask questions or suggest resources.  
* **Labels**: Applying labels (e.g., "question," "bug," "resource-suggestion," "module-1") to issues can help organize them.

### **Sharing with the Community**

Sharing the learning journey, even when it's in progress and imperfect, can be highly beneficial.

* **How to Share**: The repository URL can be shared on professional networks like LinkedIn, social media platforms like X (formerly Twitter), or relevant online forums and communities (e.g., subreddits focused on machine learning, Discord servers like Karpathy's 217).  
* **Value of Sharing**: This practice, often termed "learning in public," can attract feedback, corrections, additional resource suggestions, and even potential collaborators. It creates a form of positive accountability. Moreover, an openly documented learning process, with its inevitable challenges and breakthroughs, is often more relatable and helpful to peers than a polished end-product. This aligns with the spirit of open-source contributions, where community involvement enhances the outcome.2

### **Licensing Your Content**

Including a LICENSE file is important for clarifying how others can use the content.

* **For Code**: The MIT License is a common and permissive choice for open-source code projects.199  
* **For Written Content**: A Creative Commons license, such as CC-BY-SA 4.0 (Attribution-ShareAlike), is suitable for notes and documentation, allowing others to share and adapt the material with proper attribution, provided they share their adaptations under similar terms. This reflects the open ethos seen in datasets like FineWeb, which uses the ODC-By license.70

The act of preparing notes and structuring thoughts for potential public consumption inherently forces greater clarity and deeper understanding for the learner. Thus, sharing should be viewed not as a final step after completion, but as an integral part of the learning process itself.

## **VII. Final Thoughts and Recommendations**

Creating a GitHub repository to document a learning path and track progress is a commendable initiative that can significantly enhance the learning experience and provide value to others. The key to success lies in several areas:

1. **Clarity of Purpose**: Define clear learning goals and the scope of the journey. This will be reflected in the repository's name, description, and the structure of the README.md.  
2. **Structured Approach**: Adopt a modular approach to the learning path, breaking down complex topics into manageable units. This structure should be mirrored in both the README.md progress table and the repository's directory layout for ease of navigation.  
3. **Dynamic Documentation**: Treat the repository as a living document. Regular updates, reflections, and tracking of progress are crucial. The README.md should function as a dynamic dashboard.  
4. **Leverage Quality Resources**: The field of AI/LLMs is rich with excellent learning materials. Incorporate well-regarded courses (e.g., from Andrej Karpathy 11, DeepLearning.AI 14, Stanford 10), seminal papers 15, and practical tools (Hugging Face ecosystem 8, LM Studio 169) into the learning plan and document their use.  
5. **Embrace "Learning in Public"**: Sharing the journey, even with its imperfections, fosters personal growth through feedback and accountability, and provides a valuable resource for the wider community.  
6. **Iterative Learning**: Recognize that learning is not always linear. Be prepared to revisit foundational concepts as advanced topics are explored, and document these connections.

By following these guidelines, the user can create a well-organized, informative, and highly useful GitHub repository that not only serves as a personal learning log but also as a valuable resource for others embarking on similar educational pursuits in the exciting field of AI and Large Language Models.

#### **Works cited**

1. How The Open-Source LLM Revolution Is Transforming Enterprise AI \- Forbes, accessed May 26, 2025, [https://www.forbes.com/councils/forbestechcouncil/2025/03/20/the-open-source-llm-revolution-transforming-enterprise-ai-for-a-new-era/](https://www.forbes.com/councils/forbestechcouncil/2025/03/20/the-open-source-llm-revolution-transforming-enterprise-ai-for-a-new-era/)  
2. How Contributing to Open Source Projects Helped Me Build My Dream Career in AI \- Comet, accessed May 25, 2025, [https://www.comet.com/site/blog/contributing-to-open-source-ai/](https://www.comet.com/site/blog/contributing-to-open-source-ai/)  
3. AI News (MOVED TO news.smol.ai\!) • Buttondown, accessed May 25, 2025, [https://buttondown.com/ainews](https://buttondown.com/ainews)  
4. Andrej Karpathy, accessed May 26, 2025, [https://karpathy.ai/](https://karpathy.ai/)  
5. ReAct Prompting | Prompt Engineering Guide, accessed May 25, 2025, [https://www.promptingguide.ai/techniques/react](https://www.promptingguide.ai/techniques/react)  
6. Introduction \- Hugging Face LLM Course, accessed May 25, 2025, [https://huggingface.co/learn/llm-course/chapter1/1](https://huggingface.co/learn/llm-course/chapter1/1)  
7. Post-LLM Revolution: Exploring New Frontiers in AI Through Knowledge, Collaboration, and Co-Evolution \- BIOENGINEER.ORG, accessed May 26, 2025, [https://bioengineer.org/post-llm-revolution-exploring-new-frontiers-in-ai-through-knowledge-collaboration-and-co-evolution/](https://bioengineer.org/post-llm-revolution-exploring-new-frontiers-in-ai-through-knowledge-collaboration-and-co-evolution/)  
8. Introduction \- Hugging Face LLM Course, accessed May 26, 2025, [https://huggingface.co/learn/nlp-course/chapter1/1](https://huggingface.co/learn/nlp-course/chapter1/1)  
9. CS224n: Natural Language Processing with Deep Learning \- Stanford University, accessed May 25, 2025, [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1204/](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1204/)  
10. Stanford CS 224N | Natural Language Processing with Deep Learning, accessed May 26, 2025, [https://cs224n.stanford.edu/](https://cs224n.stanford.edu/)  
11. Neural Networks: Zero To Hero \- Andrej Karpathy, accessed May 26, 2025, [https://karpathy.ai/zero-to-hero.html](https://karpathy.ai/zero-to-hero.html)  
12. arxiv.org, accessed May 26, 2025, [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)  
13. The Illustrated Transformer – Jay Alammar – Visualizing machine ..., accessed May 26, 2025, [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)  
14. Pretraining LLMs \- DeepLearning.AI, accessed May 26, 2025, [https://www.deeplearning.ai/short-courses/pretraining-llms/](https://www.deeplearning.ai/short-courses/pretraining-llms/)  
15. arxiv.org, accessed May 26, 2025, [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)  
16. Supervised Fine-Tuning for LLMs: Step-by-Step Python Guide \- Bright Data, accessed May 25, 2025, [https://brightdata.com/blog/ai/supervised-fine-tuning](https://brightdata.com/blog/ai/supervised-fine-tuning)  
17. Finetuning Large Language Models \- DeepLearning.AI, accessed May 26, 2025, [https://www.deeplearning.ai/short-courses/finetuning-large-language-models/](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)  
18. accessed December 31, 1969, [https://www.deeplearning.ai/short-courses/llms-fine-tuning-and-merging-models/](https://www.deeplearning.ai/short-courses/llms-fine-tuning-and-merging-models/)  
19. arxiv.org, accessed May 26, 2025, [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)  
20. Llama 3, accessed May 26, 2025, [https://www.llama.com/models/llama-3/](https://www.llama.com/models/llama-3/)  
21. DeepSeek, accessed May 26, 2025, [https://www.deepseek.com/en](https://www.deepseek.com/en)  
22. What ethical concerns exist with LLMs? \- Milvus, accessed May 25, 2025, [https://milvus.io/ai-quick-reference/what-ethical-concerns-exist-with-llms](https://milvus.io/ai-quick-reference/what-ethical-concerns-exist-with-llms)  
23. accessed December 31, 1969, [https://www.deeplearning.ai/short-courses/responsible-ai-designing-ai-for-everyone/](https://www.deeplearning.ai/short-courses/responsible-ai-designing-ai-for-everyone/)  
24. Machine Learning | Coursera, accessed May 26, 2025, [https://www.coursera.org/specializations/machine-learning-introduction](https://www.coursera.org/specializations/machine-learning-introduction)  
25. What are good resources to learn math of machine learning ML for beginner students in ML and how long to have a good foundation? \- Quora, accessed May 25, 2025, [https://www.quora.com/What-are-good-resources-to-learn-math-of-machine-learning-ML-for-beginner-students-in-ML-and-how-long-to-have-a-good-foundation](https://www.quora.com/What-are-good-resources-to-learn-math-of-machine-learning-ML-for-beginner-students-in-ML-and-how-long-to-have-a-good-foundation)  
26. Linear Algebra for Machine Learning and Data Science \- Coursera, accessed May 26, 2025, [https://www.coursera.org/learn/machine-learning-linear-algebra](https://www.coursera.org/learn/machine-learning-linear-algebra)  
27. Linear Algebra | Mathematics \- MIT OpenCourseWare, accessed May 26, 2025, [https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)  
28. Calculus for Machine Learning and Data Science \- Coursera, accessed May 26, 2025, [https://www.coursera.org/learn/machine-learning-calculus](https://www.coursera.org/learn/machine-learning-calculus)  
29. Mathematics for Machine Learning: Multivariate Calculus \- Coursera, accessed May 26, 2025, [https://www.coursera.org/learn/multivariate-calculus-machine-learning](https://www.coursera.org/learn/multivariate-calculus-machine-learning)  
30. Free Course: Probability & Statistics for Machine Learning & Data Science from DeepLearning.AI | Class Central, accessed May 26, 2025, [https://www.classcentral.com/course/machine-learning-probability-and-statistics-122096](https://www.classcentral.com/course/machine-learning-probability-and-statistics-122096)  
31. Probability & Statistics for Machine Learning & Data Science \- Coursera, accessed May 26, 2025, [https://www.coursera.org/learn/machine-learning-probability-and-statistics](https://www.coursera.org/learn/machine-learning-probability-and-statistics)  
32. Deep Learning | Coursera, accessed May 26, 2025, [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning)  
33. LLM Model Training and Deep Learning Explained \- FastBots.ai, accessed May 25, 2025, [https://fastbots.ai/blog/llm-model-training-and-deep-learning-explained](https://fastbots.ai/blog/llm-model-training-and-deep-learning-explained)  
34. The Python Tutorial — Python 3.13.3 documentation, accessed May 26, 2025, [https://docs.python.org/3/tutorial/index.html](https://docs.python.org/3/tutorial/index.html)  
35. Python For Beginners | Python.org, accessed May 26, 2025, [https://www.python.org/about/gettingstarted/](https://www.python.org/about/gettingstarted/)  
36. Learn Intermediate Python 3: Object-Oriented Programming \- Codecademy, accessed May 26, 2025, [https://www.codecademy.com/learn/learn-intermediate-python-3-object-oriented-programming](https://www.codecademy.com/learn/learn-intermediate-python-3-object-oriented-programming)  
37. Learn Python 3 \- Codecademy \- GitHub, accessed May 26, 2025, [https://github.com/Codecademy/learn-python](https://github.com/Codecademy/learn-python)  
38. Python for Everybody Specialization \- Coursera, accessed May 26, 2025, [https://www.coursera.org/specializations/python](https://www.coursera.org/specializations/python)  
39. Programming for Everybody (Getting Started with Python) \- Coursera, accessed May 26, 2025, [https://www.coursera.org/programs/information-technolo-google-learning-program-ogxtk/learn/python?specialization=python](https://www.coursera.org/programs/information-technolo-google-learning-program-ogxtk/learn/python?specialization=python)  
40. NumPy User Guide, accessed May 26, 2025, [https://numpy.org/doc/2.2/numpy-user.pdf](https://numpy.org/doc/2.2/numpy-user.pdf)  
41. NumPy Tutorial: Your First Steps Into Data Science in Python, accessed May 26, 2025, [https://realpython.com/numpy-tutorial/](https://realpython.com/numpy-tutorial/)  
42. PyTorch Tutorials 2.7.0+cu126 documentation, accessed May 26, 2025, [https://docs.pytorch.org/tutorials/](https://docs.pytorch.org/tutorials/)  
43. PyTorch Tutorial for Deep Learning Researchers \- GitHub, accessed May 25, 2025, [https://github.com/yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)  
44. TensorFlow Tutorial \- GeeksforGeeks, accessed May 25, 2025, [https://www.geeksforgeeks.org/tensorflow/](https://www.geeksforgeeks.org/tensorflow/)  
45. Learn the Basics — PyTorch Tutorials 2.7.0+cu126 documentation, accessed May 26, 2025, [https://docs.pytorch.org/tutorials/beginner/basics/](https://docs.pytorch.org/tutorials/beginner/basics/)  
46. How To Use Jupyter Notebook – An Ultimate Guide \- GeeksforGeeks, accessed May 26, 2025, [https://www.geeksforgeeks.org/how-to-use-jupyter-notebook-an-ultimate-guide/](https://www.geeksforgeeks.org/how-to-use-jupyter-notebook-an-ultimate-guide/)  
47. Jupyter Notebook Tutorial for Beginners with Python \- YouTube, accessed May 26, 2025, [https://www.youtube.com/watch?v=2WL-XTl2QYI](https://www.youtube.com/watch?v=2WL-XTl2QYI)  
48. Google Colab \- A Step-by-step Guide \- AlgoTrading101 Blog, accessed May 26, 2025, [https://algotrading101.com/learn/google-colab-guide/](https://algotrading101.com/learn/google-colab-guide/)  
49. Welcome To Colab \- Colab \- Google, accessed May 26, 2025, [https://colab.research.google.com/](https://colab.research.google.com/)  
50. Jupyter Notebook: An Introduction \- Real Python, accessed May 26, 2025, [https://realpython.com/jupyter-notebook-introduction/](https://realpython.com/jupyter-notebook-introduction/)  
51. Install and Use \- Jupyter Documentation, accessed May 26, 2025, [https://docs.jupyter.org/en/stable/install.html](https://docs.jupyter.org/en/stable/install.html)  
52. Project Jupyter Documentation — Jupyter Documentation 4.1.1 ..., accessed May 26, 2025, [https://docs.jupyter.org/en/latest/index.html](https://docs.jupyter.org/en/latest/index.html)  
53. Notebook Basics — Jupyter Notebook 7.5.0a0 documentation, accessed May 26, 2025, [https://jupyter-notebook.readthedocs.io/en/latest/examples/Notebook/Notebook%20Basics.html](https://jupyter-notebook.readthedocs.io/en/latest/examples/Notebook/Notebook%20Basics.html)  
54. Attention Is All You Need \- Wikipedia, accessed May 25, 2025, [https://en.wikipedia.org/wiki/Attention\_Is\_All\_You\_Need](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need)  
55. \[2501.09166\] Attention is All You Need Until You Need Retention \- arXiv, accessed May 26, 2025, [https://arxiv.org/abs/2501.09166](https://arxiv.org/abs/2501.09166)  
56. Element-wise Attention Is All You Need \- arXiv, accessed May 26, 2025, [https://arxiv.org/pdf/2501.05730](https://arxiv.org/pdf/2501.05730)  
57. Attention Is All You Need \- arXiv, accessed May 26, 2025, [https://arxiv.org/html/1706.03762v7](https://arxiv.org/html/1706.03762v7)  
58. Element-wise Attention Is All You Need \- arXiv, accessed May 26, 2025, [https://arxiv.org/html/2501.05730v1](https://arxiv.org/html/2501.05730v1)  
59. Attention is All you Need \- NIPS, accessed May 25, 2025, [https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)  
60. You could have designed state of the art positional encoding, accessed May 25, 2025, [https://huggingface.co/blog/designing-positional-encoding](https://huggingface.co/blog/designing-positional-encoding)  
61. LLMs-from-scratch/ch05/07\_gpt\_to\_llama/converting-llama2-to-llama3.ipynb at main, accessed May 26, 2025, [https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07\_gpt\_to\_llama/converting-llama2-to-llama3.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-llama2-to-llama3.ipynb)  
62. bbycroft/llm-viz: 3D Visualization of an GPT-style LLM \- GitHub, accessed May 25, 2025, [https://github.com/bbycroft/llm-viz](https://github.com/bbycroft/llm-viz)  
63. Inside GPT: Beautiful 3D Visualization of How Language Models Actually Work \- YouTube, accessed May 25, 2025, [https://www.youtube.com/watch?v=rtcNTOvoxKY](https://www.youtube.com/watch?v=rtcNTOvoxKY)  
64. The Illustrated Transformer From Scratch, accessed May 25, 2025, [https://idtjo.hosting.acm.org/wordpress/the-illustrated-transformer/](https://idtjo.hosting.acm.org/wordpress/the-illustrated-transformer/)  
65. The Illustrated Transformer, accessed May 25, 2025, [https://the-illustrated-transformer--omosha.on.websim.ai/](https://the-illustrated-transformer--omosha.on.websim.ai/)  
66. Jay Alammar – Visualizing machine learning one concept at a time., accessed May 26, 2025, [https://jalammar.github.io/](https://jalammar.github.io/)  
67. The Illustrated Retrieval Transformer \- Jay Alammar, accessed May 26, 2025, [https://jalammar.github.io/illustrated-retrieval-transformer/](https://jalammar.github.io/illustrated-retrieval-transformer/)  
68. The Illustrated Transformer – Jay Alammar – Visualizing Machine Learning One Concept at a Time. | PDF \- Scribd, accessed May 26, 2025, [https://www.scribd.com/document/806268818/The-Illustrated-Transformer-Jay-Alammar-Visualizing-Machine-Learning-One-Concept-at-a-Time](https://www.scribd.com/document/806268818/The-Illustrated-Transformer-Jay-Alammar-Visualizing-Machine-Learning-One-Concept-at-a-Time)  
69. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale | OpenReview, accessed May 25, 2025, [https://openreview.net/forum?id=n6SCkn2QaG\&referrer=%5Bthe%20profile%20of%20Colin%20Raffel%5D(%2Fprofile%3Fid%3D\~Colin\_Raffel1)](https://openreview.net/forum?id=n6SCkn2QaG&referrer=%5Bthe+profile+of+Colin+Raffel%5D\(/profile?id%3D~Colin_Raffel1\))  
70. proceedings.neurips.cc, accessed May 26, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/370df50ccfdf8bde18f8f9c2d9151bda-Supplemental-Datasets\_and\_Benchmarks\_Track.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/370df50ccfdf8bde18f8f9c2d9151bda-Supplemental-Datasets_and_Benchmarks_Track.pdf)  
71. HuggingFaceFW/fineweb · Datasets at Hugging Face, accessed May 25, 2025, [https://huggingface.co/datasets/HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)  
72. fineweb-edu \- ModelScope, accessed May 25, 2025, [https://modelscope.cn/datasets/AI-ModelScope/fineweb-edu](https://modelscope.cn/datasets/AI-ModelScope/fineweb-edu)  
73. FineWeb Dataset \- Papers With Code, accessed May 25, 2025, [https://paperswithcode.com/dataset/fineweb](https://paperswithcode.com/dataset/fineweb)  
74. What can we learn from Hugging Face's Fineweb Dataset \- Kili Technology, accessed May 25, 2025, [https://kili-technology.com/large-language-models-llms/what-can-we-learn-from-hugging-face-s-fineweb-dataset](https://kili-technology.com/large-language-models-llms/what-can-we-learn-from-hugging-face-s-fineweb-dataset)  
75. accessed December 31, 1969, [https://huggingface.co/spaces/HuggingFaceFW/fineweb](https://huggingface.co/spaces/HuggingFaceFW/fineweb)  
76. accessed December 31, 1969, [https://huggingface.co/blog/fineweb](https://huggingface.co/blog/fineweb)  
77. openbmb/Ultra-FineWeb · Datasets at Hugging Face, accessed May 26, 2025, [https://huggingface.co/datasets/openbmb/Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb)  
78. Ultra-FineWeb: Efficient Data Filtering and Verification for High-Quality LLM Training Data, accessed May 26, 2025, [https://arxiv.org/html/2505.05427v1](https://arxiv.org/html/2505.05427v1)  
79. trendmicro-ailab/Primus-FineWeb · Datasets at Hugging Face, accessed May 26, 2025, [https://huggingface.co/datasets/trendmicro-ailab/Primus-FineWeb](https://huggingface.co/datasets/trendmicro-ailab/Primus-FineWeb)  
80. How Data Drives LLM Pretraining: Methods, Tips, and Best Practices \- Camel AI, accessed May 25, 2025, [https://www.camel-ai.org/blogs/llm-pretraining](https://www.camel-ai.org/blogs/llm-pretraining)  
81. Unveiling Challenges for LLMs in Enterprise Data Engineering \- arXiv, accessed May 25, 2025, [https://arxiv.org/html/2504.10950v1](https://arxiv.org/html/2504.10950v1)  
82. Byte-pair encoding \- Wikipedia, accessed May 26, 2025, [https://en.wikipedia.org/wiki/Byte-pair\_encoding](https://en.wikipedia.org/wiki/Byte-pair_encoding)  
83. Byte-Pair Encoding For Beginners | Towards Data Science, accessed May 26, 2025, [https://towardsdatascience.com/byte-pair-encoding-for-beginners-708d4472c0c7/](https://towardsdatascience.com/byte-pair-encoding-for-beginners-708d4472c0c7/)  
84. google/sentencepiece: Unsupervised text tokenizer for ... \- GitHub, accessed May 26, 2025, [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)  
85. Tokenization \- SentencePiece | Continuum Labs, accessed May 26, 2025, [https://training.continuumlabs.ai/training/the-fine-tuning-process/tokenization/tokenization-sentencepiece](https://training.continuumlabs.ai/training/the-fine-tuning-process/tokenization/tokenization-sentencepiece)  
86. Tiktokenizer, accessed May 26, 2025, [https://tiktokenizer.vercel.app/](https://tiktokenizer.vercel.app/)  
87. Tokenization Video Conversion | KarpathyLLMChallenge \- GitHub Pages, accessed May 25, 2025, [https://misbahsy.github.io/KarpathyLLMChallenge/TokenizationLLMChallenge.html](https://misbahsy.github.io/KarpathyLLMChallenge/TokenizationLLMChallenge.html)  
88. Build a Token Visualizer in Dotnet \- Telerik.com, accessed May 26, 2025, [https://www.telerik.com/blogs/build-token-visualizer-dotnet](https://www.telerik.com/blogs/build-token-visualizer-dotnet)  
89. Tiktokenizer vs Product XYZ \- compare the differences between ..., accessed May 26, 2025, [https://www.toolify.ai/compare/tiktokenizer-vs-product-buying-guides](https://www.toolify.ai/compare/tiktokenizer-vs-product-buying-guides)  
90. TikTokenTokenizer — TorchTune documentation, accessed May 26, 2025, [https://docs.pytorch.org/torchtune/0.1/generated/torchtune.modules.tokenizers.TikTokenTokenizer.html](https://docs.pytorch.org/torchtune/0.1/generated/torchtune.modules.tokenizers.TikTokenTokenizer.html)  
91. Practical Guide for Model Selection for Real‑World Use Cases \- OpenAI Cookbook, accessed May 26, 2025, [https://cookbook.openai.com/examples/partners/model\_selection\_guide/model\_selection\_guide](https://cookbook.openai.com/examples/partners/model_selection_guide/model_selection_guide)  
92. accessed December 31, 1969, [https\_tiktokenizer\_vercel\_app/](http://docs.google.com/https_tiktokenizer_vercel_app/)  
93. tokenization.ipynb \- GitHub Gist, accessed May 25, 2025, [https://gist.github.com/bigsnarfdude/8e99709d5c3d9d58b3831221fcbdaf68](https://gist.github.com/bigsnarfdude/8e99709d5c3d9d58b3831221fcbdaf68)  
94. LLM Token Visualizer, accessed May 25, 2025, [https://v0-llm-token-visualizer.vercel.app/](https://v0-llm-token-visualizer.vercel.app/)  
95. accessed December 31, 1969, [https://github.com/karpathy/llm.c/discussions/33](https://github.com/karpathy/llm.c/discussions/33)  
96. Andrej Karpathy's deep dive into LLMs video \- Codingscape, accessed May 25, 2025, [https://codingscape.com/blog/andrej-karpathys-deep-dive-into-llms-video](https://codingscape.com/blog/andrej-karpathys-deep-dive-into-llms-video)  
97. What is a token in AI? Understanding how AI processes language with tokenization \- Nebius, accessed May 25, 2025, [https://nebius.com/blog/posts/what-is-token-in-ai](https://nebius.com/blog/posts/what-is-token-in-ai)  
98. Understanding tokens \- .NET | Microsoft Learn, accessed May 25, 2025, [https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens)  
99. Pre-training in LLM Development \- Toloka, accessed May 25, 2025, [https://toloka.ai/blog/pre-training-in-llm-development/](https://toloka.ai/blog/pre-training-in-llm-development/)  
100. arxiv.org, accessed May 26, 2025, [https://arxiv.org/html/2401.02954v1](https://arxiv.org/html/2401.02954v1)  
101. Fine-Tuning LLMs: A Guide With Examples \- DataCamp, accessed May 25, 2025, [https://www.datacamp.com/tutorial/fine-tuning-large-language-models](https://www.datacamp.com/tutorial/fine-tuning-large-language-models)  
102. Supervised Fine-Tuning vs. RLHF: How to Choose the Right Approach to Train Your LLM, accessed May 25, 2025, [https://www.invisible.co/blog/supervised-fine-tuning-vs-rlhf-how-to-choose-the-right-approach-to-train-your-llm](https://www.invisible.co/blog/supervised-fine-tuning-vs-rlhf-how-to-choose-the-right-approach-to-train-your-llm)  
103. Supervised Fine-Tuning Vs RLHF Vs RL: Differences and Role in LLMs \- Nudgebee, accessed May 25, 2025, [https://nudgebee.com/blog/supervised-fine-tuning-rlhf-rl-comparison-llms.html](https://nudgebee.com/blog/supervised-fine-tuning-rlhf-rl-comparison-llms.html)  
104. Understanding LLM Training Data: A Comprehensive Guide \- Uniphore, accessed May 25, 2025, [https://www.uniphore.com/glossary/llm-training-data/](https://www.uniphore.com/glossary/llm-training-data/)  
105. What Is Reinforcement Learning From Human Feedback (RLHF ..., accessed May 25, 2025, [https://www.ibm.com/think/topics/rlhf](https://www.ibm.com/think/topics/rlhf)  
106. Reinforcement learning from human feedback \- Wikipedia, accessed May 25, 2025, [https://en.wikipedia.org/wiki/Reinforcement\_learning\_from\_human\_feedback](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback)  
107. Reinforcement Learning from Human Feedback \- DeepLearning.AI, accessed May 26, 2025, [https://www.deeplearning.ai/short-courses/reinforcement-learning-from-human-feedback/](https://www.deeplearning.ai/short-courses/reinforcement-learning-from-human-feedback/)  
108. An Introduction to Model Merging for LLMs | NVIDIA Technical Blog, accessed May 25, 2025, [https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/](https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/)  
109. PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation \- arXiv, accessed May 25, 2025, [https://arxiv.org/html/2504.18583](https://arxiv.org/html/2504.18583)  
110. Decoding and Search Strategies \- ἐντελέχεια.άι, accessed May 25, 2025, [https://lecture.jeju.ai/lectures/nlp\_deep/llms/decoding.html](https://lecture.jeju.ai/lectures/nlp_deep/llms/decoding.html)  
111. Decoding Strategies: How LLMs Choose The Next Word \- AssemblyAI, accessed May 25, 2025, [https://www.assemblyai.com/blog/decoding-strategies-how-llms-choose-the-next-word](https://www.assemblyai.com/blog/decoding-strategies-how-llms-choose-the-next-word)  
112. How to generate text: using different decoding methods for language ..., accessed May 26, 2025, [https://huggingface.co/blog/how-to-generate](https://huggingface.co/blog/how-to-generate)  
113. Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies \- arXiv, accessed May 26, 2025, [https://www.arxiv.org/pdf/2502.05202](https://www.arxiv.org/pdf/2502.05202)  
114. Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs \- arXiv, accessed May 26, 2025, [https://arxiv.org/html/2407.01082v5](https://arxiv.org/html/2407.01082v5)  
115. What is LLM Temperature? \- Hopsworks, accessed May 25, 2025, [https://www.hopsworks.ai/dictionary/llm-temperature](https://www.hopsworks.ai/dictionary/llm-temperature)  
116. accessed December 31, 1969, [https://www.deeplearning.ai/short-courses/building-evaluating-llms/](https://www.deeplearning.ai/short-courses/building-evaluating-llms/)  
117. Chatbot Arena (formerly LMSYS): Free AI Chat to Compare & Test ..., accessed May 26, 2025, [https://lmarena.ai/](https://lmarena.ai/)  
118. \[2501.13106\] VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding \- arXiv, accessed May 25, 2025, [https://arxiv.org/abs/2501.13106](https://arxiv.org/abs/2501.13106)  
119. \[2407.21783\] The Llama 3 Herd of Models \- ar5iv \- arXiv, accessed May 25, 2025, [https://ar5iv.labs.arxiv.org/html/2407.21783](https://ar5iv.labs.arxiv.org/html/2407.21783)  
120. Introducing Llama 3.1: Our most capable models to date \- Meta AI, accessed May 25, 2025, [https://ai.meta.com/blog/meta-llama-3-1/](https://ai.meta.com/blog/meta-llama-3-1/)  
121. Introducing Meta Llama 3: The most capable openly available LLM to date, accessed May 25, 2025, [https://ai.meta.com/blog/meta-llama-3/](https://ai.meta.com/blog/meta-llama-3/)  
122. The Power of Meta AI Llama 3: Advancements in AI Technology \- Vision Computer Solutions, accessed May 26, 2025, [https://www.vcsolutions.com/blog/unleash-the-power-of-meta-ai-llama-3/](https://www.vcsolutions.com/blog/unleash-the-power-of-meta-ai-llama-3/)  
123. arxiv.org, accessed May 26, 2025, [https://arxiv.org/pdf/2404.14047](https://arxiv.org/pdf/2404.14047)  
124. arXiv:2411.07133v3 \[cs.AI\] 26 Feb 2025, accessed May 26, 2025, [https://arxiv.org/pdf/2411.07133?](https://arxiv.org/pdf/2411.07133)  
125. Llama 3.2 Meta's New generation Models Vertex AI | Google Cloud Blog, accessed May 26, 2025, [https://cloud.google.com/blog/products/ai-machine-learning/llama-3-2-metas-new-generation-models-vertex-ai](https://cloud.google.com/blog/products/ai-machine-learning/llama-3-2-metas-new-generation-models-vertex-ai)  
126. What If We Recaption Billions of Web Images with LLaMA-3? \- arXiv, accessed May 26, 2025, [https://arxiv.org/html/2406.08478v2](https://arxiv.org/html/2406.08478v2)  
127. Llama 3 \- Klu.ai, accessed May 26, 2025, [https://klu.ai/glossary/llama-3](https://klu.ai/glossary/llama-3)  
128. Llama 3 in Action: Deployment Strategies and Advanced Functionality for Real-World Applications \- InfoQ, accessed May 26, 2025, [https://www.infoq.com/articles/llama3-deployment-applications/](https://www.infoq.com/articles/llama3-deployment-applications/)  
129. llama3/MODEL\_CARD.md at main · meta-llama/llama3 \- GitHub, accessed May 26, 2025, [https://github.com/meta-llama/llama3/blob/main/MODEL\_CARD.md](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)  
130. accessed December 31, 1969, [https\_arxiv\_org/abs/2401.02954](http://docs.google.com/https_arxiv_org/abs/2401.02954)  
131. arxiv.org, accessed May 25, 2025, [https://arxiv.org/abs/2401.02954](https://arxiv.org/abs/2401.02954)  
132. \[2504.03665\] LLM & HPC:Benchmarking DeepSeek's Performance in High-Performance Computing Tasks \- arXiv, accessed May 25, 2025, [https://arxiv.org/abs/2504.03665](https://arxiv.org/abs/2504.03665)  
133. DeepSeek-R2: China's Powerful New AI Model for 2025, accessed May 26, 2025, [https://deepseek.ai/blog/deepseek-r2-ai-model-launch-2025](https://deepseek.ai/blog/deepseek-r2-ai-model-launch-2025)  
134. What is DeepSeek? A full breakdown of the disruptive open-source LLM \- Tenable, accessed May 25, 2025, [https://www.tenable.com/cybersecurity-guide/learn/deepseek-ai-guide](https://www.tenable.com/cybersecurity-guide/learn/deepseek-ai-guide)  
135. Customize DeepSeek-R1 671b model using Amazon SageMaker HyperPod recipes – Part 2, accessed May 25, 2025, [https://aws.amazon.com/blogs/machine-learning/customize-deepseek-r1-671b-model-using-amazon-sagemaker-hyperpod-recipes-part-2/](https://aws.amazon.com/blogs/machine-learning/customize-deepseek-r1-671b-model-using-amazon-sagemaker-hyperpod-recipes-part-2/)  
136. DeepSeek-V3 \+ SGLang: Inference Optimization — Blog \- DataCrunch, accessed May 25, 2025, [https://datacrunch.io/blog/deepseek-v3-sglang-inference-optimization](https://datacrunch.io/blog/deepseek-v3-sglang-inference-optimization)  
137. accessed December 31, 1969, [https://huggingface.co/blog/deepseek-llm](https://huggingface.co/blog/deepseek-llm)  
138. arxiv.org, accessed May 26, 2025, [https://arxiv.org/html/2401.14196v1](https://arxiv.org/html/2401.14196v1)  
139. arxiv.org, accessed May 26, 2025, [https://arxiv.org/pdf/2405.04434](https://arxiv.org/pdf/2405.04434)  
140. DeepSeek, accessed May 26, 2025, [https://chat.deepseek.com/](https://chat.deepseek.com/)  
141. DeepSeek Platform, accessed May 26, 2025, [https://platform.deepseek.com/](https://platform.deepseek.com/)  
142. DeepSeek AI Introduces CODEI/O: A Novel Approach that Transforms Code-based Reasoning Patterns into Natural Language Formats to Enhance LLMs' Reasoning Capabilities \- Reddit, accessed May 26, 2025, [https://www.reddit.com/r/machinelearningnews/comments/1iqdpet/deepseek\_ai\_introduces\_codeio\_a\_novel\_approach/](https://www.reddit.com/r/machinelearningnews/comments/1iqdpet/deepseek_ai_introduces_codeio_a_novel_approach/)  
143. \[2504.21801\] DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition \- arXiv, accessed May 26, 2025, [https://arxiv.org/abs/2504.21801](https://arxiv.org/abs/2504.21801)  
144. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning \- arXiv, accessed May 26, 2025, [https://arxiv.org/html/2501.12948v1](https://arxiv.org/html/2501.12948v1)  
145. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning \- arXiv, accessed May 26, 2025, [https://arxiv.org/pdf/2501.12948](https://arxiv.org/pdf/2501.12948)  
146. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model \- GitHub, accessed May 26, 2025, [https://github.com/deepseek-ai/DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)  
147. \[2504.07128\] DeepSeek-R1 Thoughtology: Let's think about LLM Reasoning \- arXiv, accessed May 26, 2025, [https://arxiv.org/abs/2504.07128](https://arxiv.org/abs/2504.07128)  
148. R1dacted: Investigating Local Censorship in DeepSeek's R1 Language Model \- arXiv, accessed May 26, 2025, [https://arxiv.org/html/2505.12625v1](https://arxiv.org/html/2505.12625v1)  
149. DeepSeek: Everything you need to know about this new LLM in one place \- Daily.dev, accessed May 26, 2025, [https://daily.dev/blog/deepseek-everything-you-need-to-know-about-this-new-llm-in-one-place](https://daily.dev/blog/deepseek-everything-you-need-to-know-about-this-new-llm-in-one-place)  
150. Announcing DeepSeek-R1 in private preview on Snowflake Cortex AI, accessed May 26, 2025, [https://www.snowflake.com/en/blog/deepseek-preview-snowflake-cortex-ai/](https://www.snowflake.com/en/blog/deepseek-preview-snowflake-cortex-ai/)  
151. DeepSeek: Everything You Need To Know \- Neontri, accessed May 26, 2025, [https://neontri.com/blog/deepseek-features-and-risks/](https://neontri.com/blog/deepseek-features-and-risks/)  
152. DeepSeek V3.1: The New Frontier in Artificial Intelligence, accessed May 26, 2025, [https://deepseek.ai/blog/deepseek-v31](https://deepseek.ai/blog/deepseek-v31)  
153. Google, OpenAI, and Meta — AI leaderboard jockeying heats up, the grim forecast of “AI 2027,” and the shadow of looming chip tariffs | Center for Security and Emerging Technology, accessed May 25, 2025, [https://cset.georgetown.edu/newsletter/april-24-2025/](https://cset.georgetown.edu/newsletter/april-24-2025/)  
154. Large Language Models: What You Need to Know in 2025 | HatchWorks AI, accessed May 25, 2025, [https://hatchworks.com/blog/gen-ai/large-language-models-guide/](https://hatchworks.com/blog/gen-ai/large-language-models-guide/)  
155. Top 9 Large Language Models as of May 2025 | Shakudo, accessed May 25, 2025, [https://www.shakudo.io/blog/top-9-large-language-models](https://www.shakudo.io/blog/top-9-large-language-models)  
156. accessed December 31, 1969, [https\_huggingface\_co/spaces/huggingface-projects/huggingface-inference-playground](http://docs.google.com/https_huggingface_co/spaces/huggingface-projects/huggingface-inference-playground)  
157. accessed December 31, 1969, [https://huggingface.co/spaces/huggingface-projects/huggingface-inference-playground](https://huggingface.co/spaces/huggingface-projects/huggingface-inference-playground)  
158. huggingface/inference-playground \- GitHub, accessed May 25, 2025, [https://github.com/huggingface/inference-playground](https://github.com/huggingface/inference-playground)  
159. Transformers, what can they do? \- Hugging Face LLM Course, accessed May 25, 2025, [https://huggingface.co/learn/llm-course/chapter1/3](https://huggingface.co/learn/llm-course/chapter1/3)  
160. Transformers.js \- Hugging Face, accessed May 25, 2025, [https://huggingface.co/docs/transformers.js/index](https://huggingface.co/docs/transformers.js/index)  
161. Hub Integration \- Hugging Face, accessed May 25, 2025, [https://huggingface.co/docs/inference-providers/hub-integration](https://huggingface.co/docs/inference-providers/hub-integration)  
162. accessed December 31, 1969, [https://www.deeplearning.ai/short-courses/open-source-models-with-hugging-face/](https://www.deeplearning.ai/short-courses/open-source-models-with-hugging-face/)  
163. Building Generative AI Applications with Gradio \- DeepLearning.AI, accessed May 26, 2025, [https://www.deeplearning.ai/short-courses/building-generative-ai-applications-with-gradio/](https://www.deeplearning.ai/short-courses/building-generative-ai-applications-with-gradio/)  
164. Transformers \- Hugging Face, accessed May 26, 2025, [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)  
165. Datasets \- Hugging Face, accessed May 26, 2025, [https://huggingface.co/docs/datasets/index](https://huggingface.co/docs/datasets/index)  
166. Tokenizers \- Hugging Face, accessed May 26, 2025, [https://huggingface.co/docs/tokenizers/index](https://huggingface.co/docs/tokenizers/index)  
167. accessed December 31, 1969, [https://huggingface.co/spaces/huggingface/text-generation-inference](https://huggingface.co/spaces/huggingface/text-generation-inference)  
168. accessed December 31, 1969, [https\_lmstudio\_ai/](http://docs.google.com/https_lmstudio_ai/)  
169. LM Studio \- Discover, download, and run local LLMs, accessed May 26, 2025, [https://lmstudio.ai/](https://lmstudio.ai/)  
170. LMStudio \- Run ANY Open-Source Large Language Models Locally \- YouTube, accessed May 25, 2025, [https://www.youtube.com/watch?v=0x8yk8ACBMo](https://www.youtube.com/watch?v=0x8yk8ACBMo)  
171. api.together.ai, accessed May 26, 2025, [https://api.together.xyz/playground](https://api.together.xyz/playground)  
172. Together AI LLM integration guide \- LiveKit Docs, accessed May 25, 2025, [https://docs.livekit.io/agents/integrations/llm/together/](https://docs.livekit.io/agents/integrations/llm/together/)  
173. Function calling \- Introduction \- Together AI, accessed May 25, 2025, [https://docs.together.ai/docs/function-calling](https://docs.together.ai/docs/function-calling)  
174. Hyperbolic GPU Marketplace: On-Demand NVIDIA GPU Rentals ..., accessed May 26, 2025, [https://app.hyperbolic.xyz/](https://app.hyperbolic.xyz/)  
175. Deep Dive Into Hyperbolic's Serverless Inference Service, accessed May 25, 2025, [https://hyperbolic.xyz/blog/deep-dive-into-hyperbolic-inference](https://hyperbolic.xyz/blog/deep-dive-into-hyperbolic-inference)  
176. Top AI Inference Providers \- Hyperbolic, accessed May 25, 2025, [https://hyperbolic.xyz/blog/top-ai-inference-providers](https://hyperbolic.xyz/blog/top-ai-inference-providers)  
177. Understanding Reasoning LLMs \- Sebastian Raschka, accessed May 25, 2025, [https://sebastianraschka.com/blog/2025/understanding-reasoning-llms.html](https://sebastianraschka.com/blog/2025/understanding-reasoning-llms.html)  
178. StateAct: Enhancing LLM Base Agents via Self-prompting and State-tracking \- arXiv, accessed May 25, 2025, [https://arxiv.org/html/2410.02810v3](https://arxiv.org/html/2410.02810v3)  
179. Evaluating the thinking process of reasoning LLMs : r/datascience \- Reddit, accessed May 25, 2025, [https://www.reddit.com/r/datascience/comments/1imkowl/evaluating\_the\_thinking\_process\_of\_reasoning\_llms/](https://www.reddit.com/r/datascience/comments/1imkowl/evaluating_the_thinking_process_of_reasoning_llms/)  
180. LLM Powered Autonomous Agents | Lil'Log, accessed May 26, 2025, [https://lilianweng.github.io/posts/2023-06-23-agent/](https://lilianweng.github.io/posts/2023-06-23-agent/)  
181. Line of Duty: Evaluating LLM Self-Knowledge via Consistency in Feasibility Boundaries \- ACL Anthology, accessed May 25, 2025, [https://aclanthology.org/2025.trustnlp-main.10.pdf](https://aclanthology.org/2025.trustnlp-main.10.pdf)  
182. TELL ME ABOUT YOURSELF: LLMS ARE AWARE OF THEIR LEARNED BEHAVIORS, accessed May 25, 2025, [https://martins1612.github.io/selfaware\_paper\_betley.pdf](https://martins1612.github.io/selfaware_paper_betley.pdf)  
183. What are the challenges in ensuring the LLM relies on the retrieved information rather than its parametric knowledge? How might we evaluate if the model is “cheating” by using memorized info? \- Milvus, accessed May 25, 2025, [https://milvus.io/ai-quick-reference/what-are-the-challenges-in-ensuring-the-llm-relies-on-the-retrieved-information-rather-than-its-parametric-knowledge-how-might-we-evaluate-if-the-model-is-cheating-by-using-memorized-info](https://milvus.io/ai-quick-reference/what-are-the-challenges-in-ensuring-the-llm-relies-on-the-retrieved-information-rather-than-its-parametric-knowledge-how-might-we-evaluate-if-the-model-is-cheating-by-using-memorized-info)  
184. Tell me about yourself: LLMs are aware of their learned behaviors \- LessWrong, accessed May 25, 2025, [https://www.lesswrong.com/posts/xrv2fNJtqabN3h6Aj/tell-me-about-yourself-llms-are-aware-of-their-learned](https://www.lesswrong.com/posts/xrv2fNJtqabN3h6Aj/tell-me-about-yourself-llms-are-aware-of-their-learned)  
185. AIs are becoming more self-aware. Here's why that matters \- AI Digest, accessed May 25, 2025, [https://theaidigest.org/self-awareness](https://theaidigest.org/self-awareness)  
186. Theory of Mind Imitation by LLMs for Physician-Like Human Evaluation \- medRxiv, accessed May 25, 2025, [https://www.medrxiv.org/content/10.1101/2025.03.01.25323142v2.full.pdf](https://www.medrxiv.org/content/10.1101/2025.03.01.25323142v2.full.pdf)  
187. www.medrxiv.org, accessed May 25, 2025, [https://www.medrxiv.org/content/10.1101/2025.03.01.25323142v1.full.pdf](https://www.medrxiv.org/content/10.1101/2025.03.01.25323142v1.full.pdf)  
188. What Role Does Memory Play in the Performance of LLMs? \- Association of Data Scientists, accessed May 25, 2025, [https://adasci.org/what-role-does-memory-play-in-the-performance-of-llms/](https://adasci.org/what-role-does-memory-play-in-the-performance-of-llms/)  
189. accessed December 31, 1969, [https://news.mit.edu/2024/can-large-language-models-help-us-understand-ourselves-0123](https://news.mit.edu/2024/can-large-language-models-help-us-understand-ourselves-0123)  
190. Salesforce AI Research Details Agentic Advancements \- Salesforce, accessed May 26, 2025, [https://www.salesforce.com/news/stories/ai-research-agentic-advancements/](https://www.salesforce.com/news/stories/ai-research-agentic-advancements/)  
191. Jagged Intelligence Is What's Wrong With Enterprise AI According to Salesforce, accessed May 25, 2025, [https://aimresearch.co/market-industry/jagged-intelligence-is-whats-wrong-with-enterprise-ai-according-to-salesforce](https://aimresearch.co/market-industry/jagged-intelligence-is-whats-wrong-with-enterprise-ai-according-to-salesforce)  
192. "Researchers have come up with a term to describe AI's pattern of reasoning: “jagged intelligence.” It refers to the fact that, as Andrej Karpathy explained, state-of-the-art AI models “can both perform extremely impressive tasks while simultaneously struggling with some very dumb problems.” \#AI" — Bluesky, accessed May 25, 2025, [https://bsky.app/profile/ai-everyday.bsky.social/post/3ljzsoiksuc2e](https://bsky.app/profile/ai-everyday.bsky.social/post/3ljzsoiksuc2e)  
193. Andrej Karpathy Coined a New Term 'Jagged Intelligence': Understanding the Inconsistencies in Advanced AI \- MarkTechPost, accessed May 25, 2025, [https://www.marktechpost.com/2024/08/11/andrej-karpathy-coined-a-new-term-jagged-intelligence-understanding-the-inconsistencies-in-advanced-ai/](https://www.marktechpost.com/2024/08/11/andrej-karpathy-coined-a-new-term-jagged-intelligence-understanding-the-inconsistencies-in-advanced-ai/)  
194. accessed December 31, 1969, [https://www.salesforceairesearch.com/research/understanding-the-jagged-frontier-of-large-language-models](https://www.salesforceairesearch.com/research/understanding-the-jagged-frontier-of-large-language-models)  
195. Salesforce AI Research Team on Combating AI Hallucinations ..., accessed May 26, 2025, [https://www.salesforce.com/news/stories/combating-ai-hallucinations/](https://www.salesforce.com/news/stories/combating-ai-hallucinations/)  
196. Data Challenges in Enhancing LLM Quality and AI Performance \- RIM-AI, accessed May 25, 2025, [https://www.rim-ai.com/blog/data-challenges-llm](https://www.rim-ai.com/blog/data-challenges-llm)  
197. Ethical considerations of generative AI \- NTT Data, accessed May 26, 2025, [https://www.nttdata.com/global/en/-/media/nttdataglobal/1\_files/insights/reports/generative-ai/ethical-considerations-of-genai/ethical-considerations-of-generative-ai.pdf](https://www.nttdata.com/global/en/-/media/nttdataglobal/1_files/insights/reports/generative-ai/ethical-considerations-of-genai/ethical-considerations-of-generative-ai.pdf)  
198. Ethical Concerns of Generative AI and Mitigation Strategies: A Systematic Mapping Study \- arXiv, accessed May 26, 2025, [https://arxiv.org/pdf/2502.00015](https://arxiv.org/pdf/2502.00015)  
199. karpathy/nn-zero-to-hero: Neural Networks \- GitHub, accessed May 26, 2025, [https://github.com/karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)  
200. chizkidd/Karpathy-Neural-Networks-Zero-to-Hero \- GitHub, accessed May 26, 2025, [https://github.com/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero](https://github.com/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero)  
201. karpathy/llm.c: LLM training in simple, raw C/CUDA \- GitHub, accessed May 26, 2025, [https://github.com/karpathy/llm.c](https://github.com/karpathy/llm.c)  
202. Andrej Karpathy \- GitHub, accessed May 26, 2025, [https://github.com/karpathy](https://github.com/karpathy)  
203. ChatGPT Prompt Engineering for Developers \- DeepLearning.AI, accessed May 26, 2025, [https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)  
204. LangChain for LLM Application Development \- DeepLearning.AI, accessed May 26, 2025, [https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)  
205. LangChain: Chat with Your Data \- DeepLearning.AI, accessed May 26, 2025, [https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/)  
206. accessed December 31, 1969, [https://www.deeplearning.ai/short-courses/mastering-llm-ops-at-scale/](https://www.deeplearning.ai/short-courses/mastering-llm-ops-at-scale/)  
207. Red Teaming LLM Applications \- DeepLearning.AI, accessed May 26, 2025, [https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/)  
208. Stanford CS224N Natural Language Processing with Deep Learning I Spring 2024 I Professor Christopher Manning \- YouTube, accessed May 25, 2025, [https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D)  
209. fastai/course-nlp: A Code-First Introduction to NLP course \- GitHub, accessed May 25, 2025, [https://github.com/fastai/course-nlp](https://github.com/fastai/course-nlp)  
210. fast.ai Code-First Intro to Natural Language Processing \- YouTube, accessed May 25, 2025, [https://www.youtube.com/playlist?list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9](https://www.youtube.com/playlist?list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9)  
211. accessed December 31, 1969, [https://www.fast.ai/posts/2022-10-18-nlp-with-transformers.html](https://www.fast.ai/posts/2022-10-18-nlp-with-transformers.html)  
212. HuggingFaceFW/fineweb-edu · Datasets at Hugging Face, accessed May 25, 2025, [https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer)  
213. AlphaGo Zero: Starting from scratch \- Google DeepMind, accessed May 25, 2025, [https://deepmind.google/discover/blog/alphago-zero-starting-from-scratch/](https://deepmind.google/discover/blog/alphago-zero-starting-from-scratch/)  
214. AlphaGo: Mastering the Game of Go with Deep Neural Networks and Tree Search, accessed May 26, 2025, [https://kam.mff.cuni.cz/\~hladik/OS/Slides/Ha-GO-abs-2016.pdf](https://kam.mff.cuni.cz/~hladik/OS/Slides/Ha-GO-abs-2016.pdf)  
215. Mastering the game of Go with deep neural networks and tree search \- Google Research, accessed May 26, 2025, [https://research.google/pubs/mastering-the-game-of-go-with-deep-neural-networks-and-tree-search/](https://research.google/pubs/mastering-the-game-of-go-with-deep-neural-networks-and-tree-search/)  
216. Where can I find the alphago "paper"? : r/baduk \- Reddit, accessed May 25, 2025, [https://www.reddit.com/r/baduk/comments/499mg1/where\_can\_i\_find\_the\_alphago\_paper/](https://www.reddit.com/r/baduk/comments/499mg1/where_can_i_find_the_alphago_paper/)  
217. Let's build GPT: from scratch, in code, spelled out by Andrej Karpathy \[video\] | Hacker News, accessed May 25, 2025, [https://news.ycombinator.com/item?id=34414716](https://news.ycombinator.com/item?id=34414716)  
218. karpathy/minbpe: Minimal, clean code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. \- GitHub, accessed May 26, 2025, [https://github.com/karpathy/minbpe](https://github.com/karpathy/minbpe)  
219. Chain rule (video) \- Khan Academy, accessed May 26, 2025, [https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-2-new/ab-3-1a/v/chain-rule-introduction](https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-2-new/ab-3-1a/v/chain-rule-introduction)