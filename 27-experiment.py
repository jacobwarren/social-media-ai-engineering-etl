from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import argparse
from utils.logging_setup import init_pipeline_logging
from utils.run_id import get_last_run_id
from utils.manifest import read_manifest, discover_input

logger = init_pipeline_logging("phase3.experiment", None, "27-experiment")

parser = argparse.ArgumentParser(description="Experiment with a trained model")
parser.add_argument("--run-id", default=None)
parser.add_argument("--base-dir", default="data/processed")
parser.add_argument("--model", default=None, help="Override model path")
parser.add_argument("--stage", default="auto", choices=["auto","sft","grpo"], help="Which trained model to use")
args, _ = parser.parse_known_args()

if args.run_id == "latest":
    latest = get_last_run_id(args.base_dir)
    if latest:
        args.run_id = latest

model_path = args.model
if args.run_id and not model_path:
    try:
        m = read_manifest(args.run_id, args.base_dir)
        if args.stage in ("auto","grpo"):
            found = discover_input(m, "26-train-grpo")
            if found:
                model_path = found[0] if isinstance(found, list) else found
        if not model_path and args.stage in ("auto","sft"):
            found = discover_input(m, "25-train-sft")
            if found:
                model_path = found[0] if isinstance(found, list) else found
    except Exception:
        pass

if not model_path:
    logger.warning("No model discovered; falling back to base HF model")
    model_path = "Qwen/Qwen2.5-7B-Instruct"

# Sample prompts.
prompts = [
    """
<|im_start|>user
# Request
Create a LinkedIn post that **inspires and motivates** **on the topic of**: `Entrepreneurship`

### Key Message
```
I went from having almost no money in the bank to building a multi-million pound supplement company. I discuss the traits needed for success, the point when you should stop learning and start earning, and the marketing strategies that helped my company grow rapidly. Alan Barratt of Grenade built a multi-million pound supplement company starting from next to no cash in the bank. The episode of Sam's Business Growth Show covers his journey, the entrepreneurial traits required, the transition from learning to earning, and the initial marketing strategies that enabled rapid sales and growth. Tune in to the podcast links provided for more details.
```
### Writing Constraints
- **Suggested Post Length**: Between 750 and 1,500 characters long
- **Emoji Usage**: high
- **Tone**: witty, inspiring, engaging
### Writing Style Features
- **Sentence Structure**: A mix of short and long sentences, showing a balanced style.
- **Vocabulary Usage**: A rich vocabulary, showcasing extensive language use and depth.
- **Line Break Usage**: Frequent line breaks, contributing to an easy-to-read structure.
- **Punctuation**: Regular use of periods.
- **Bullet Styles**: Multiple bullet styles detected, indicating a creative or varied structuring.
- **Topic Shifts**: Consistent topic focus, highlighting a thorough exploration of a single subject.
- **Narrative Flow**: A complex narrative flow that progresses through multiple stages, indicating a dynamic and layered storytelling approach.
- **Pacing**: The pacing is described as 'Moderate', indicating the rhythm and speed of the narrative.
- **Sentiment Arc**: A diverse emotional range, showcasing a dynamic and unpredictable sentiment.
- **Profanity Level**: none<|im_end|>
<|im_start|>assistant
""",
    """<|im_start|>user
# Request
Create a LinkedIn post that **reflects on an experience** **on the topic of**: `Networking and Professional Development`

### Key Message
```
Networking is a crucial skill, and I designed a 100 day challenge to help people grow their network and cybersecurity skills. The challenge involves posting something new every day for 100 days, tagging a friend on the first day, and including a specific hashtag. This challenge is for everyone, regardless of experience level, and is aimed at helping participants get out of their comfort zone and gain more exposure to recruiters and hiring managers on LinkedIn. 85% of jobs are filled through networking, according to HubSpot. Designed a 100 day challenge to help grow network and cybersecurity skills. Challenge requires posting daily on LinkedIn with #cybertechdave100daysofcyberchallenge, tagging a friend on Day 1, and includes no experience requirement. Aimed at helping participants gain exposure to recruiters and hiring managers on LinkedIn.
```
### Writing Constraints
- **Suggested Post Length**: Between 1,500 and 3,000 characters long
- **Emoji Usage**: high
- **Tone**: informative, exciting, assertive
### Writing Style Features
- **Sentence Structure**: Long and complex sentences, indicating a detailed and elaborate style.
- **Vocabulary Usage**: A rich vocabulary, showcasing extensive language use and depth.
- **Common Phrases**: sign-off: caring 😉 #infosecurity #linkedin #connections #network #share #comment, brand: charles sturt university, greeting: ❓ do you want to gain 50 #cybersecurity, brand: the diana initiative, brand: david meece
- **Section Divider**: `*`
- **Line Break Usage**: Frequent line breaks, contributing to an easy-to-read structure.
- **Punctuation**: Standard punctuation usage.
- **Bullet Styles**: Uses varying emojis as bullet points, adding a casual and modern touch.
- **Topic Shifts**: Consistent topic focus, highlighting a thorough exploration of a single subject.
- **Narrative Flow**: A complex narrative flow that progresses through multiple stages, indicating a dynamic and layered storytelling approach.
- **Pacing**: The pacing is described as 'Slow', indicating the rhythm and speed of the narrative.
- **Sentiment Arc**: A diverse emotional range, showcasing a dynamic and unpredictable sentiment.
- **Profanity Level**: none<|im_end|>
<|im_start|>assistant
""",
    """<|im_start|>user
# Request
Create a LinkedIn post that **challenges popular opinions** **on the topic of**: `Coding And Development`

### Key Message
```
I would learn git to revert mistakes, focus on finishing projects by starting small, and improve communication with my team on code and designs. But above all, I would choose to start learning now. Learn git to revert mistakes by reading Pro Git, Atlassian Git, and starting a GitHub. Focus on finishing projects by building front-end, front-end/backend, and full-stack apps. Improve communication by taking public speaking courses, learning to present, and empathizing with the audience. The above is a small list of how I wish I could learn how to code again. But if I had to choose one thing above anything else, I would choose to start learning now.
```
### Writing Constraints
- **Suggested Post Length**: Between 750 and 1,500 characters long
- **Emoji Usage**: none
- **Tone**: caring, informative, inspiring
### Writing Style Features
- **Sentence Structure**: Long and complex sentences, indicating a detailed and elaborate style.
- **Vocabulary Usage**: A rich vocabulary, showcasing extensive language use and depth.
- **Common Phrases**: sign-off: #alexcancode #softwareengineer #coding
- **Line Break Usage**: Frequent line breaks, contributing to an easy-to-read structure.
- **Punctuation**: Regular use of periods.
- **Bullet Styles**: Multiple bullet styles detected, indicating a creative or varied structuring.
- **Topic Shifts**: Consistent topic focus, highlighting a thorough exploration of a single subject.
- **Narrative Flow**: A complex narrative flow that progresses through multiple stages, indicating a dynamic and layered storytelling approach.
- **Pacing**: The pacing is described as 'Slow', indicating the rhythm and speed of the narrative.
- **Sentiment Arc**: A diverse emotional range, showcasing a dynamic and unpredictable sentiment.
- **Profanity Level**: none<|im_end|>
<|im_start|>assistant
""",
    """<|im_start|>user
# Request
Create a LinkedIn post that **shares a step-by-step guide** **on the topic of**: `Public Relations`

### Key Message
```
PR can help my business by providing local and national exposure, improving SEO with backlinks, and driving website traffic. To do this myself, I craft a press release from a creative and compelling story, target a strategic list of journalists with personalized pitches, follow up effectively, and ensure my website has a lead generation system in place. Craft a press release from a creative and compelling story - ideas include charity campaigns, unique actions, events, overcoming challenges, or hitting milestones. Curate a strategic journalist list by targeting niche, local, and nationwide journalists. Pitch well with personalization and creativity, follow up for links, and ensure a lead gen system on your website.
```
### Writing Constraints
- **Suggested Post Length**: Between 750 and 1,500 characters long
- **Emoji Usage**: medium
- **Tone**: informative, assertive, engaging
### Writing Style Features
- **Sentence Structure**: Long and complex sentences, indicating a detailed and elaborate style.
- **Vocabulary Usage**: A rich vocabulary, showcasing extensive language use and depth.
- **Common Phrases**: brand: nancy doyle, greeting: this pr campaign i worked on only went, sign-off: doing research that impacts others & helps them.
- **Line Break Usage**: Frequent line breaks, contributing to an easy-to-read structure.
- **Punctuation**: Standard punctuation usage.
- **Bullet Styles**: Multiple bullet styles detected, indicating a creative or varied structuring.
- **Topic Shifts**: Consistent topic focus, highlighting a thorough exploration of a single subject.
- **Narrative Flow**: The narrative flow includes: introducing the subject; then Content; then Content; then Content.
- **Pacing**: The pacing is described as 'Slow', indicating the rhythm and speed of the narrative.
- **Sentiment Arc**: A diverse emotional range, showcasing a dynamic and unpredictable sentiment.
- **Profanity Level**: none<|im_end|>
<|im_start|>assistant
""",
    """<|im_start|>user
# Request
Create a LinkedIn post that **inspires and motivates** **on the topic of**: `DevOps Training and Projects`

### Key Message
```
I believe that working on real-world DevOps projects is essential for both entry-level and advanced-level DevOps engineers to gain practical experience and stand out in job interviews. Doing these projects will give you the implementation knowledge needed to impress recruiters and measure up with someone with 3 years of experience. Certifications alone are not enough; you need to have worked on production-related projects to secure a job over other candidates. I also teach how to document projects using GitHub, which is a valuable skill for any DevOps engineer. I have 26 real-time projects that will provide 3 years of work experience, broaden knowledge, and prepare DevOps engineers for jobs. Recruiters seek candidates with production-related projects, not just certifications. I teach GitHub markdown and project documentation. These projects help stand out in interviews and match the experience of a 3-year veteran. Send a message if interested.
```
### Writing Constraints
- **Suggested Post Length**: Between 1,500 and 3,000 characters long
- **Emoji Usage**: medium
- **Tone**: caring, assertive, upbeat
### Writing Style Features
- **Sentence Structure**: Long and complex sentences, indicating a detailed and elaborate style.
- **Vocabulary Usage**: A rich vocabulary, showcasing extensive language use and depth.
- **Common Phrases**: brand: amazon web services, brand: cloud computing, brand: microsoft azure, brand: azure basics, brand: amazon aws s3, brand: getting started, brand: deploy azure virtual desktop, brand: training for beginners, brand: git & github, sign-off: #happylearning #serverlesscomputing #deploy #amazonaws #beginners #manage #yourcareer #basics, sign-off: #cicd #cloud9 #cloudformation #ec2 #linux #pythonprogramming #python #pythonlearning, brand: quick start, brand: docker tutorial, brand: git crash course, brand: free aws
- **Line Break Usage**: Frequent line breaks, contributing to an easy-to-read structure.
- **Punctuation**: Standard punctuation usage.
- **Bullet Styles**: Multiple bullet styles detected, indicating a creative or varied structuring.
- **Topic Shifts**: Consistent topic focus, highlighting a thorough exploration of a single subject.
- **Narrative Flow**: A complex narrative flow that progresses through multiple stages, indicating a dynamic and layered storytelling approach.
- **Pacing**: The pacing is described as 'Slow', indicating the rhythm and speed of the narrative.
- **Sentiment Arc**: A diverse emotional range, showcasing a dynamic and unpredictable sentiment.
- **Profanity Level**: none<|im_end|>
<|im_start|>assistant
""",
    """<|im_start|>user
# Request
Create a LinkedIn post that **offers keen observations** **on the topic of**: `Digital Marketing Strategy`

### Key Message
```
Sometimes a channel just doesn't work, and it was hard for me to accept that Google Ads, which I relied on for capturing existing demand, wasn't converting qualified leads into sales despite high CTRs and conversion rates. After months of burning through budget with little return, I had to shut it down and learn not to be too attached to any one channel or strategy, always ready to cut losses if the important signals indicate it's not working. Google Ads was my go-to for capturing existing demand. CTRs were well above 4%, conversion rates were sometimes higher than CTRs, and most leads were sales qualified. Over 6 months, the campaign burned through budget with very little to show for it. So, we turned it off.
```
### Writing Constraints
- **Suggested Post Length**: Between 750 and 1,500 characters long
- **Emoji Usage**: high
- **Tone**: casual, informative, rebellious
### Writing Style Features
- **Sentence Structure**: A mix of short and long sentences, showing a balanced style.
- **Vocabulary Usage**: A rich vocabulary, showcasing extensive language use and depth.
- **Common Phrases**: brand: chris walker, greeting: "gate or no gate for a white paper, greeting: why don't companies talk to their customers in, sign-off: #b2bmarketing #industrialmarketing #gatedcontent #contentstrategy, sign-off: #customerresearch #b2bmarketing #marketingstrategy #industrialmarketing #careerdevelopment
- **Line Break Usage**: Frequent line breaks, contributing to an easy-to-read structure.
- **Punctuation**: Standard punctuation usage.
- **Bullet Styles**: Uses varying emojis as bullet points, adding a casual and modern touch.
- **Topic Shifts**: Consistent topic focus, highlighting a thorough exploration of a single subject.
- **Narrative Flow**: A complex narrative flow that progresses through multiple stages, indicating a dynamic and layered storytelling approach.
- **Pacing**: The pacing is described as 'Moderate', indicating the rhythm and speed of the narrative.
- **Sentiment Arc**: A diverse emotional range, showcasing a dynamic and unpredictable sentiment.
- **Profanity Level**: none<|im_end|>
<|im_start|>assistant
""",
    """<|im_start|>user
# Request
Create a LinkedIn post that **shares a step-by-step guide** **on the topic of**: `Content Creation`

### Key Message
```
I find that the key to building a successful digital business is using my brand to tell a unique story that no one else can tell. By identifying a problem, solving it, sharing my solution, and then packaging it to sell, I position myself as a creator-of-one, eliminating competition and standing out in the creator economy. Most people can see a problem, but only a small percentage actually solve it, share their solution, and monetize it, making those who do part of the top 1% of creators. By identifying a problem, solving it, sharing my solution, and then packaging it to sell, I position myself as a creator-of-one, eliminating competition and standing out in the creator economy. Most people can see a problem, but only a small percentage actually solve it, share their solution, and monetize it, making those who do part of the top 1% of creators.
```
### Writing Constraints
- **Suggested Post Length**: Between 1,500 and 3,000 characters long
- **Emoji Usage**: low
- **Tone**: witty, persuasive, inspiring
### Writing Style Features
- **Sentence Structure**: Long and complex sentences, indicating a detailed and elaborate style.
- **Vocabulary Usage**: A rich vocabulary, showcasing extensive language use and depth.
- **Common Phrases**: brand: p.s. writing, sign-off: to profitable creator join 10,000+ others reading: jonbrosio.com/curve, sign-off: unlock one 6-figure solo business tip, every week., brand: create curve, sign-off: to unlock your 6-figure solo business, every week., brand: the waffle house marketing strategy
- **Line Break Usage**: Frequent line breaks, contributing to an easy-to-read structure.
- **Punctuation**: Regular use of periods.
- **Bullet Styles**: Multiple bullet styles detected, indicating a creative or varied structuring.
- **Topic Shifts**: Consistent topic focus, highlighting a thorough exploration of a single subject.
- **Narrative Flow**: A complex narrative flow that progresses through multiple stages, indicating a dynamic and layered storytelling approach.
- **Pacing**: The pacing is described as 'Slow', indicating the rhythm and speed of the narrative.
- **Sentiment Arc**: A diverse emotional range, showcasing a dynamic and unpredictable sentiment.
- **Profanity Level**: none<|im_end|>
<|im_start|>assistant
""",
    """<|im_start|>user
# Request
Create a LinkedIn post that **offers keen observations** **on the topic of**: `Professional Communication and Problem Solving`

### Key Message
```
I find that when I answer questions at work without addressing the deeper concerns, people start doubting my competence. To improve responses, I focus on the 'question behind the question,' ensuring I provide a thorough and reassuring answer that addresses their underlying concerns. This approach prevents repetitive questioning, skepticism, and the need for endless back-and-forth, saving time and emotional labor. By understanding what the person truly values and cares about, I can tailor my response to make them feel reassured and safe, even if they don't know exactly what they're asking. Improving your communication skills are just one of the topics we cover in the Maven Course Accelerator. It's a 2-week program on how to teach an engaging, live course. It's free. Apply here: https://bit.ly/3PNrYvC
```
### Writing Constraints
- **Suggested Post Length**: Between 1,500 and 3,000 characters long
- **Emoji Usage**: low
- **Tone**: informative, persuasive, professional
### Writing Style Features
- **Sentence Structure**: A mix of short and long sentences, showing a balanced style.
- **Vocabulary Usage**: A rich vocabulary, showcasing extensive language use and depth.
- **Line Break Usage**: Frequent line breaks, contributing to an easy-to-read structure.
- **Punctuation**: Regular use of periods.
- **Bullet Styles**: Uses varying emojis as bullet points, adding a casual and modern touch.
- **Topic Shifts**: Consistent topic focus, highlighting a thorough exploration of a single subject.
- **Narrative Flow**: A complex narrative flow that progresses through multiple stages, indicating a dynamic and layered storytelling approach.
- **Pacing**: The pacing is described as 'Moderate', indicating the rhythm and speed of the narrative.
- **Sentiment Arc**: A diverse emotional range, showcasing a dynamic and unpredictable sentiment.
- **Profanity Level**: none<|im_end|>
<|im_start|>assistant
""",
    """<|im_start|>user
# Request
Create a LinkedIn post that **challenges popular opinions** **on the topic of**: `Personal Branding`

### Key Message
```
I believe that every scientific invention or discovery has already been discussed in our Vedic culture. Spirituality helps us realize that we are one with the universe, and it's 'Adiyogi' who introduced these spiritual concepts. The concept of 'Quantum Entanglement' further supports the wisdom our saints have given us in the form of spirituality. Every scientific invention or discovery has already been discussed in our Vedic culture. Spirituality makes us realise that we are one with the universe. We don't have proof of who introduced the spiritual concepts in humans but we know it's 'Adiyogi'. This concept of 'Quantum Entanglement' will let us think once more of what our Saints have given us in the form of 'Spirituality'. Video credit- Instagram
```
### Writing Constraints
- **Suggested Post Length**: Up to 750 characters long
- **Emoji Usage**: none
- **Tone**: witty, serious, inspiring
### Writing Style Features
- **Sentence Structure**: A mix of short and long sentences, showing a balanced style.
- **Vocabulary Usage**: A rich vocabulary, showcasing extensive language use and depth.
- **Common Phrases**: sign-off: #castudents #icaistudsnts #caaspirants #caarticleship #cafinalist #cafinal #caintermediate #cafoundation, brand: 𝐰𝐚𝐲𝐬. sometimes, brand: 𝐲𝐨𝐮𝐫 𝐞𝐲𝐞𝐬, brand: 𝗶𝘀 𝗻𝗼𝘁 𝗼𝗻𝗹𝘆 𝗮𝗯𝗼𝘂𝘁, brand: 𝗼𝗿 𝗼𝗻𝗹𝘆 𝗮𝗯𝗼𝘂𝘁, greeting: sangita, don't know how you will take it, sign-off: people ❤️🙏 #godspresence #life #experiences #learnings #ca #spirituality
- **Line Break Usage**: Frequent line breaks, contributing to an easy-to-read structure.
- **Punctuation**: Standard punctuation usage.
- **Bullet Styles**: Uses • for bullet points, indicating a structured format.
- **Topic Shifts**: Consistent topic focus, highlighting a thorough exploration of a single subject.
- **Narrative Flow**: A complex narrative flow that progresses through multiple stages, indicating a dynamic and layered storytelling approach.
- **Pacing**: The pacing is described as 'Moderate', indicating the rhythm and speed of the narrative.
- **Sentiment Arc**: A diverse emotional range, showcasing a dynamic and unpredictable sentiment.
- **Profanity Level**: none<|im_end|>
<|im_start|>assistant
"""
]
# Create a sampling params object.
sampling_params = SamplingParams(
    max_tokens=8192,
    temperature=1.5,
    min_p=0.1
)

# Create an LLM.
llm = LLM(model=model_path)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(
    prompts, 
    sampling_params,
)
# Print the outputs.
print("\nGenerated Outputs:\n" + "-" * 60)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"""Prompt:

{prompt}

""")
    print(f"""Output:

{generated_text}

""")
    print("-" * 60)