import { LMStudioClient } from "@lmstudio/sdk";
import mammoth from 'mammoth';


const client = new LMStudioClient();

const documentPath = '/Users/zoedauphinee/hackathon/hackaton/Working Code/pdf-1050g-sample_rental_agreement_basic_beginning_renting_an_apartment_or_house.docx';
const modelPath = 'mlx-community/Llama-3.2-3B-Instruct-4bit';


async function main(question: string) {

  const { value: documentText } = await mammoth.extractRawText({ path: documentPath });

  const llama = await client.llm.load(modelPath, {identifier: 'my-model'});

  const prediction = llama.respond([
    { role: "system", content: documentText },
    { role: "user", content: question },
  ]);

  for await (const { content } of prediction) {
    process.stdout.write(content);
  }

  const { stats } = await prediction;
  console.log(stats);

  await client.llm.unload('my-model');
}


const question = "What is the penalty for a late payment?";
main(question).catch(console.error);
