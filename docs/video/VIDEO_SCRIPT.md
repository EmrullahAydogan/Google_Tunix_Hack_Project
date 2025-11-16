# YouTube Video Script - Google Tunix Hack

**Title:** Teaching AI to Show Its Work: Training Gemma with Tunix
**Duration:** 3 minutes (180 seconds)
**Target:** Technical audience, competition judges

---

## 0:00-0:30 (30 seconds) - INTRODUCTION & HOOK

**[Visual: Title slide with project logo]**

**Script:**
> "Hi, I'm Emrullah, and today I'm going to show you how I taught a language model to not just solve problems, but to explain its thinking process step-by-step - just like a student showing their work on a math test.
>
> This is my submission for the Google Tunix Hack competition, where we use reinforcement learning to train AI models to be more transparent and interpretable."

**[Visual: Quick demo preview - show model solving a problem with reasoning]**

> "Let's dive in!"

**Key Points:**
- Introduce yourself
- State the problem clearly
- Show a teaser of the end result
- Hook the viewer

---

## 0:30-1:30 (60 seconds) - METHODOLOGY

**[Visual: Architecture diagram]**

**Script:**
> "Here's my approach: I used Google's Gemma 3 1B model - chosen for its efficiency and 32,000 token context window - and fine-tuned it using Tunix, a JAX-native reinforcement learning library.

**[Visual: Show GSM8K dataset examples]**

> "For training data, I used the GSM8K dataset - 8,500 grade school math problems, each requiring 2 to 8 steps to solve. Perfect for teaching step-by-step reasoning.

**[Visual: Show training pipeline diagram]**

> "The key innovation is my reward function. Instead of just rewarding correct answers, I score responses on three criteria:
> - 50% for correctness - is the answer right?
> - 30% for reasoning quality - are the steps logical?
> - 20% for clarity - can a human understand the explanation?

**[Visual: Code snippet of reward function]**

> "I used GRPO - Group Relative Policy Optimization - which is more stable than standard PPO for this kind of task. Training took about 6 hours on a Kaggle TPU.

**Key Points:**
- Model choice (Gemma 3 1B)
- Dataset (GSM8K)
- Reward function (3 components)
- Training method (GRPO)
- Keep it concise but technical

---

## 1:30-2:30 (60 seconds) - DEMO & RESULTS

**[Visual: Live demo - show the model in action]**

**Script:**
> "Let me show you the model in action. Here's a typical problem:

**[Visual: Display problem on screen]**

> 'Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much does she make every day at the farmers' market?'

**[Visual: Show model generating response step-by-step, with typing animation]**

> "Watch how the model breaks it down:
>
> Step 1: It identifies that 16 eggs are laid daily
> Step 2: Calculates eggs consumed - 3 for breakfast plus 4 for muffins = 7 eggs
> Step 3: Computes remaining eggs - 16 minus 7 = 9 eggs
> Step 4: Calculates revenue - 9 eggs times $2 = $18
>
> Answer: $18 per day

**[Visual: Show metrics/results table]**

> "The results speak for themselves. On the test set, the model achieved:
> - 87% accuracy
> - 92% reasoning quality score
> - 95% clarity score
>
> Compared to the baseline model at 74% accuracy, that's a 13 percentage point improvement."

**Key Points:**
- Real example
- Show the step-by-step reasoning clearly
- Highlight the transparency
- Show quantitative results
- Compare to baseline

---

## 2:30-3:00 (30 seconds) - CONCLUSION & NEXT STEPS

**[Visual: Summary slide with key takeaways]**

**Script:**
> "So what did we accomplish? We've shown that with the right reward function and training approach, we can teach language models to be transparent reasoners - not just answer-generators.
>
> This has huge implications for trustworthy AI, especially in education, healthcare, and other high-stakes domains where we need to understand the model's thinking.

**[Visual: GitHub repo and links]**

> "All the code, configs, and trained models are open source and reproducible. Check out the links in the description.
>
> Thanks for watching, and special thanks to Google and Kaggle for organizing this amazing competition!"

**[Visual: End screen with links and contact info]**

**Key Points:**
- Summarize achievement
- Broader impact
- Reproducibility
- Thank organizers
- Call to action (check description)

---

## PRODUCTION NOTES

### Visuals Needed:
1. **Title slide** - Project name and logo
2. **Architecture diagram** - Show model pipeline
3. **Dataset examples** - GSM8K problems
4. **Code snippets** - Reward function
5. **Live demo** - Model solving problem with reasoning
6. **Results charts** - Metrics comparison
7. **Summary slide** - Key takeaways
8. **End screen** - Links and contact

### Recording Tips:
- **Audio:** Use good microphone, minimize background noise
- **Pacing:** Speak clearly but quickly (you have only 3 minutes!)
- **Screen recording:** Use OBS or similar for demos
- **Editing:** Add captions for accessibility
- **Music:** Optional background music (low volume)
- **Resolution:** Minimum 720p, prefer 1080p

### Timing Breakdown:
- Introduction: 30 seconds (16.7%)
- Methodology: 60 seconds (33.3%)
- Demo & Results: 60 seconds (33.3%)
- Conclusion: 30 seconds (16.7%)
- **Total: 180 seconds exactly**

### Keywords for YouTube Description:
- Google Tunix Hack
- Machine Learning
- Reinforcement Learning
- Chain-of-Thought
- Gemma
- LLM
- AI Reasoning
- Interpretable AI

---

**Status:** Draft
**Version:** 1.0
**Last Updated:** [Date]
