// Import required libraries
import dotenv from 'dotenv';
dotenv.config();

import fs from "fs";
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RetrievalQAChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// Debug log
console.log('Environment variables loaded:', {
    API_KEY: process.env.API_KEY ? 'Present' : 'Missing',
    MODEL: process.env.MODEL,
    PURPOSE: process.env.PURPOSE,
    DOCUMENT_CONTENT: process.env.DOCUMENT_CONTENT ? 'Present' : 'Missing'
});

// Initialize with environment variables
const GOOGLE_API_KEY = process.env.API_KEY;
if (!GOOGLE_API_KEY) {
    throw new Error("API_KEY environment variable is required");
}

const text = process.env.DOCUMENT_CONTENT || "No content provided";
const model = process.env.MODEL || "gemini-pro";
const embedding_model = process.env.EMBEDDING_MODEL;

// History
let chatHistory = [];

// Refined system prompt logic
const purpose = process.env.PURPOSE || "general";
let refinedPrompt = "You are a helpful assistant.";

switch (purpose) {
    case "personal-assistant":
        refinedPrompt = "You are a highly organized personal assistant who helps users manage tasks, schedules, and personal information.";
        break;
    case "education":
        refinedPrompt = "You are a knowledgeable educational tutor assisting students with learning, explaining concepts clearly and simply.";
        break;
    case "e-commerce":
        refinedPrompt = "You are a smart e-commerce assistant helping users find products, track orders, and compare options.";
        break;
    case "customer-support":
        refinedPrompt = "You are a professional and friendly customer support representative addressing customer queries and resolving issues efficiently.";
        break;
    default:
        refinedPrompt = "You are a helpful assistant.";
        break;
}

let chain;
let initialized = false;

async function initializeChat() {
    if (initialized) return;

    if (text === "No content provided") {
        throw new Error("DOCUMENT_CONTENT environment variable is required");
    }

    try {
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
            separators: ["\n\n", "\n", " ", ""],
        });

        const docs = await splitter.createDocuments([text.replace(/'''/g, '')]);

        const embeddings = new GoogleGenerativeAIEmbeddings({
            modelName: embedding_model,
            apiKey: GOOGLE_API_KEY,
        });

        const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
        const retriever = vectorStore.asRetriever();

        const llm = new ChatGoogleGenerativeAI({
            modelName: model,
            apiKey: GOOGLE_API_KEY,
        });

        const template = refinedPrompt + `

Context:
{context}

Conversation History:
{history}

Visitor's Question:
{question}

Response:
`;

        const prompt = PromptTemplate.fromTemplate(template);

        chain = await RetrievalQAChain.fromLLM(llm, retriever, {
            returnSourceDocuments: false,
            prompt,
        });

        initialized = true;
    } catch (error) {
        console.error('Error initializing chat:', error);
        initialized = false;
        throw error;
    }
}

// Main message processing function with history
export async function processUserMessage(message) {
    if (!initialized) {
        await initializeChat();
    }

    // Update chat history
    chatHistory.push({ user: message });

    try {
        const formattedHistory = chatHistory
            .map(entry => entry.user ? `User: ${entry.user}` : `AI: ${entry.ai}`)
            .slice(-6) // Limit to last 6 exchanges to avoid context overload
            .join('\n');

        const response = await chain.invoke({
            query: message,
            history: formattedHistory,
        });

        chatHistory[chatHistory.length - 1].ai = response.text;

        return response.text;
    } catch (error) {
        console.error('Error processing message:', error);
        return "I apologize, but I encountered an error processing your request. Please try again.";
    }
}
