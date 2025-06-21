import panel as pn
from ctransformers import AutoModelForCausalLM
import asyncio
from threading import Thread
from queue import Queue
import re
import base64
import os
from pathlib import Path

# Initialize Panel
pn.extension()

class VamshisAIAssistant:
    def __init__(self):
        # Model configuration
        self.model = None
        self.model_loaded = False
        self.loading = False
        
        # Custom avatar (base64 encoded placeholder)
        self.avatar = "vamshi_image.jpg"
        
        # Pre-loaded knowledge about Vamshi
        self.personal_knowledge = {
            "interests": ["Artificial Intelligence", "Quantum Computing", "Classical Music", "Hiking"],
            "background": "Computer Science graduate with specialization in Machine Learning from a top university. Currently working on advanced AI systems.",
            "projects": [
                "Personalized Learning Assistant: An AI system that adapts to individual learning styles",
                "Quantum ML Algorithms: Developing hybrid quantum-classical machine learning models"
            ],
            "preferences": {
                "Programming Language": "Python",
                "Music Composer": "Ludwig van Beethoven",
                "Hiking Location": "Swiss Alps",
                "Research Focus": "Explainable AI and Quantum Neural Networks"
            }
        }
        
        # Load personal knowledge files
        self.knowledge_files = {}
        knowledge_dir = Path("personal_knowledge")
        if knowledge_dir.exists():
            for file in knowledge_dir.glob("*.txt"):
                with open(file, "r", encoding="utf-8") as f:
                    self.knowledge_files[file.name] = f.read()
        
        # Conversation settings
        self.conversation_history = []
        self.max_history = 8
        
        # Special commands
        self.special_commands = {
            "/projects": "Show Vamshi's current projects",
            "/interests": "List Vamshi's main interests",
            "/background": "Show professional background",
            "/prefs": "Show personal preferences",
            "/files": "List available knowledge files",
            "/file [filename]": "Show contents of a specific knowledge file",
            "/clear": "Clear conversation history",
            "/load": "Initialize the AI engine"
        }
        
        # Create chat interface with compatible parameters
        self.chat_interface = pn.chat.ChatInterface(
            callback=self._callback,
            user="Vamshi's AI",
            avatar=self.avatar,
            show_clear=False,
            sizing_mode="stretch_width",
            height=700,
            styles={
                'background': '#f8f9fa',
                'border': '1px solid #dee2e6',
                'border-radius': '10px',
                'padding': '10px'
            }
        )
        
        # Custom controls
        self.status = pn.widgets.StaticText(
            value="⚪ Model not loaded",
            styles={'font-weight': 'bold', 'color': '#4f46e5'}
        )
        self.temp_slider = pn.widgets.FloatSlider(
            name="Creativity", 
            value=0.7, 
            start=0.1, 
            end=1.5, 
            step=0.1,
            styles={'color': '#4f46e5'}
        )
        self.max_tokens = pn.widgets.IntSlider(
            name="Response Length", 
            value=512, 
            start=64, 
            end=2048, 
            step=64,
            styles={'color': '#4f46e5'}
        )
        
        # Response queue for async processing
        self.response_queue = Queue()
        
        # Send welcome message
        self._send_welcome_message()

    def _send_welcome_message(self):
        """Send personalized welcome message"""
        welcome_msg = f"""
        <div style='background:#4f46e5;color:white;padding:20px;border-radius:10px'>
        <h2>👋 Welcome to Your Personal AI Assistant, Vamshi!</h2>
        <p>I'm your customized AI powered by Mistral 7B with knowledge about:</p>
        <ul>
        <li>Your professional background in <strong>{self.personal_knowledge['background'].split()[0]} {self.personal_knowledge['background'].split()[1]}</strong></li>
        <li>Your current projects: <strong>{', '.join([p.split(':')[0] for p in self.personal_knowledge['projects']])}</strong></li>
        <li>Your personal preferences and interests</li>
        </ul>
        
        <h3>Special Commands:</h3>
        <div style='columns: 2;'>
        {''.join([f"<div><code>{cmd}</code>: {desc}</div>" for cmd, desc in self.special_commands.items()])}
        </div>
        
        <p style='margin-top:15px;'>Type <strong>/load</strong> to initialize the AI engine (5-10 min first time)</p>
        </div>
        """
        self.chat_interface.send(welcome_msg, user="System", respond=False)

    def _load_model(self):
        """Load the Mistral 7B model with CPU-only configuration"""
        if self.model_loaded or self.loading:
            return
            
        self.loading = True
        self.status.value = "🟡 Loading AI engine (5-10 minutes)..."
        
        loading_msg = """
        <div style='background:#f0f4ff;padding:15px;border-radius:8px'>
        <h3>⚙️ Initializing Vamshi's Personal AI Assistant</h3>
        <p>Downloading and loading the Mistral 7B model (4.1GB)...</p>
        <ul>
        <li>This only happens once</li>
        <li>Subsequent starts will be faster</li>
        <li>Working entirely on CPU</li>
        </ul>
        </div>
        """
        self.chat_interface.send(loading_msg, user="System", respond=False)
        
        try:
            # Load the model with CPU-only configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                model_type="mistral",
                gpu_layers=0,  # Force CPU-only
                threads=8,     # Use 8 CPU threads
                context_length=4096,
                batch_size=1   # Lower batch size for CPU
            )
            
            self.model_loaded = True
            self.status.value = "🟢 AI Engine Ready"
            
            ready_msg = f"""
            <div style='background:#e6f7e6;padding:15px;border-radius:8px'>
            <h3>✅ Vamshi's Personal AI Assistant is Ready!</h3>
            <p>Now with enhanced knowledge about:</p>
            <ul>
            <li>Your <strong>{len(self.personal_knowledge['interests'])} core interests</strong></li>
            <li>Your <strong>{len(self.personal_knowledge['projects'])} active projects</strong></li>
            <li><strong>{len(self.knowledge_files)} personal knowledge files</strong> integrated</li>
            </ul>
            <p>How can I assist you today?</p>
            </div>
            """
            self.chat_interface.send(ready_msg, user="System", respond=False)
            
        except Exception as e:
            error_msg = f"""
            <div style='background:#ffebee;padding:15px;border-radius:8px'>
            <h3>❌ Error Loading AI Engine</h3>
            <p><strong>Error:</strong> {str(e)}</p>
            <p>Possible solutions:</p>
            <ul>
            <li>Check internet connection</li>
            <li>Ensure you have 5GB+ disk space</li>
            <li>Try again later</li>
            </ul>
            </div>
            """
            self.chat_interface.send(error_msg, user="System", respond=False)
            self.status.value = f"🔴 Error: {str(e)}"
        finally:
            self.loading = False

    def _format_prompt(self, new_input):
        """Format the conversation history into a proper instruction prompt with personal context"""
        # Personal context header
        prompt = f"""
        [VAMSHI'S PERSONAL CONTEXT]
        Background: {self.personal_knowledge['background']}
        Current Interests: {', '.join(self.personal_knowledge['interests'])}
        Active Projects: {', '.join([p.split(':')[0] for p in self.personal_knowledge['projects']])}
        Personal Preferences: {', '.join([f"{k} ({v})" for k,v in self.personal_knowledge['preferences'].items()])}
        
        [CONVERSATION HISTORY]
        """
        
        # Add previous exchanges
        for exchange in self.conversation_history[-self.max_history:]:
            role = exchange["role"]
            content = exchange["content"]
            if role == "user":
                prompt += f"User: {content}\n"
            else:
                prompt += f"Assistant: {content}\n"
        
        # Add the new input
        prompt += f"""
        [INSTRUCTION]
        Using Vamshi's personal context and conversation history, provide a helpful, detailed response to:
        User: {new_input}
        Assistant: """
        
        # Clean up the prompt
        prompt = re.sub(r'\n\s+', '\n', prompt).strip()
        return prompt

    def _generate_response(self, prompt):
        """Generate response optimized for CPU with personal context"""
        try:
            if not self.model_loaded:
                return "AI engine not loaded. Type /load to initialize."
                
            # Format prompt with history and personal context
            formatted_prompt = self._format_prompt(prompt)
            
            # Generate response with CPU-optimized parameters
            response = self.model(
                formatted_prompt,
                max_new_tokens=self.max_tokens.value,
                temperature=self.temp_slider.value,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.15,
                stop=["User:", "Assistant:"],
                stream=False
            )
            
            # Clean response
            response = response.split("Assistant:")[-1].strip()
            response = response.split("[INSTRUCTION]")[0].strip()
            return response
            
        except Exception as e:
            return f"⚠️ Error generating response: {str(e)}"

    async def _callback(self, contents: str, user: str, instance: pn.chat.ChatInterface):
        """Handle user input with special commands and conversation"""
        # Handle special commands
        if contents.lower() == "/projects":
            projects = "\n".join([f"• {p}" for p in self.personal_knowledge["projects"]])
            return f"🔨 Vamshi's Current Projects:\n{projects}"
            
        elif contents.lower() == "/interests":
            interests = "\n".join([f"• {i}" for i in self.personal_knowledge["interests"]])
            return f"🎯 Vamshi's Main Interests:\n{interests}"
            
        elif contents.lower() == "/background":
            return f"📚 Professional Background:\n{self.personal_knowledge['background']}"
            
        elif contents.lower() == "/prefs":
            prefs = "\n".join([f"• {k}: {v}" for k,v in self.personal_knowledge["preferences"].items()])
            return f"❤️ Personal Preferences:\n{prefs}"
            
        elif contents.lower() == "/files":
            if not self.knowledge_files:
                return "📂 No personal knowledge files found in 'personal_knowledge' folder"
            files = "\n".join([f"• {f}" for f in self.knowledge_files.keys()])
            return f"📂 Available Knowledge Files:\n{files}\n\nUse '/file filename.txt' to view contents"
            
        elif contents.lower().startswith("/file "):
            filename = contents[6:].strip()
            if filename in self.knowledge_files:
                return f"📄 Contents of {filename}:\n\n{self.knowledge_files[filename][:2000]}..." \
                       if len(self.knowledge_files[filename]) > 2000 else self.knowledge_files[filename]
            return f"❌ File {filename} not found. Use '/files' to list available files."
            
        elif contents.lower() == "/load":
            if not self.model_loaded and not self.loading:
                Thread(target=self._load_model, daemon=True).start()
                return "🔄 Starting AI engine initialization..."
            return "ℹ️ AI engine is already loading or loaded"
            
        elif contents.lower() == "/clear":
            self.conversation_history = []
            instance.objects = []
            self._send_welcome_message()
            return
            
        elif contents.lower().startswith("/"):
            commands = "\n".join([f"{cmd}: {desc}" for cmd, desc in self.special_commands.items()])
            return f"❓ Unknown command. Available special commands:\n{commands}"
        
        # Normal conversation handling
        if not self.model_loaded:
            return "⚠️ Please type /load to initialize the AI engine first (5-10 min first time)"
            
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": contents})
        
        # Show "typing" indicator
        instance.placeholder = "Vamshi's AI is thinking..."
        
        # Run generation in a separate thread
        def generate_and_put():
            response = self._generate_response(contents)
            self.response_queue.put(response)
            
        Thread(target=generate_and_put, daemon=True).start()
        
        # Wait for response
        while self.response_queue.empty():
            await asyncio.sleep(0.1)
            
        # Get response and clear queue
        response = self.response_queue.get()
        self.response_queue.queue.clear()
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Clear typing indicator
        instance.placeholder = ""
        
        return response

    def view(self):
        """Return the complete Panel view with all components"""
        return pn.Column(
            pn.pane.Markdown(
                """
                <div style='background:#4f46e5;color:white;padding:15px;border-radius:10px'>
                <h1 style='margin:0;'>Vamshi's Personal AI Assistant</h1>
                <p style='margin:0;'>Powered by Mistral 7B with Custom Knowledge Integration</p>
                </div>
                """,
                sizing_mode="stretch_width"
            ),
            pn.Row(
                self.status,
                pn.layout.HSpacer(),
                self.temp_slider,
                self.max_tokens,
                styles={
                    'background': '#f8fafc',
                    'padding': '10px',
                    'border-radius': '8px',
                    'align-items': 'center'
                },
                sizing_mode="stretch_width"
            ),
            self.chat_interface,
            pn.pane.Markdown(
                """
                <div style='background:#f0f4ff;padding:15px;border-radius:8px'>
                <h3 style='margin-top:0;'>💡 Assistant Features</h3>
                <ul>
                <li><strong>Personal Context:</strong> Knows about your projects, interests, and preferences</li>
                <li><strong>File Integration:</strong> Loads .txt files from 'personal_knowledge' folder</li>
                <li><strong>Special Commands:</strong> Quick access to your personal information</li>
                <li><strong>CPU-Optimized:</strong> Runs without GPU using efficient quantization</li>
                </ul>
                </div>
                """,
                sizing_mode="stretch_width"
            ),
            styles={
                'background': '#ffffff',
                'padding': '15px',
                'border-radius': '12px',
                'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
            },
            sizing_mode="stretch_width",
            height=800
        )

# Create and serve the assistant
assistant = VamshisAIAssistant()
assistant.view().servable()