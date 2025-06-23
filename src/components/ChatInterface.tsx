import React, { useState, useRef, useEffect } from 'react';
import { AppState, ChatMessage, UserInfo, Chat } from '../types';
import { Send, FileUp, User, Bot } from 'lucide-react';
import ChatBubble from './ChatBubble';
import FileUpload from './FileUpload';
 
interface ChatInterfaceProps {
  userInfo: UserInfo;
  chats: Chat[];
  activeChat: string | null;
  isTyping: boolean;
  setState: React.Dispatch<React.SetStateAction<AppState>>;
  sessionId: string | null;
  user_Id: string | null;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  userInfo,
  chats,
  activeChat,
  isTyping,
  setState,
  sessionId,
  user_Id,
}) => {
  const [input, setInput] = useState('');
  const [showFileUpload, setShowFileUpload] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const sessionReady = Boolean(sessionId && user_Id);
  const activeMessages = chats.find(chat => chat.id === activeChat)?.messages || [];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeMessages]);

  // Remove the duplicate session creation logic from ChatInterface
  // Session should only be created in App.tsx or UserInfoForm.tsx

  const handleSendMessage = async () => {
    if (!input.trim() || !activeChat || !sessionReady) return;

    const now = new Date();
    const newUserMessage: ChatMessage = {
      type: 'user',
      content: input.trim(),
      timestamp: now,
    };

    setState(prev => ({
      ...prev,
      chats: prev.chats.map(chat =>
        chat.id === activeChat
          ? { ...chat, messages: [...chat.messages, newUserMessage] }
          : chat
      ),
      isTyping: true,
    }));

    setInput('');

    try {
      const response = await fetch('http://127.0.0.1:8000/multimodal_agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_message: newUserMessage.content,
          chat_history: activeMessages.map(msg => ({
            sender: msg.type,
            text: msg.content,
          })),
          session_id: sessionId,
          user_id: user_Id,
        }),
      });

      if (!response.ok) throw new Error('Failed to get response');
      const data = await response.json();

      const botResponse: ChatMessage = {
        type: 'assistant',
        content: data.reply || "Sorry, I couldn't process your request. You can contact the operation team regarding this query at hello@stylework.city",
        timestamp: data.timestamp ? new Date(data.timestamp) : new Date(),
      };

      setState(prev => ({
        ...prev,
        chats: prev.chats.map(chat =>
          chat.id === activeChat
            ? { ...chat, messages: [...chat.messages, botResponse] }
            : chat
        ),
        isTyping: false,
      }));
    } catch (error) {
      console.error('Chat error:', error);

      setState(prev => ({
        ...prev,
        chats: prev.chats.map(chat =>
          chat.id === activeChat
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  {
                    type: 'assistant',
                    content: "Sorry, I couldn't process your request at the moment. You can contact the operation team regarding this query at hello@stylework.city",
                    timestamp: new Date(),
                  },
                ],
              }
            : chat
        ),
        isTyping: false,
      }));
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleFileUpload = (files: File[]) => {
    if (files.length > 0 && activeChat) {
      const fileNames = files.map(file => file.name).join(', ');
      const newUserMessage: ChatMessage = {
        type: 'user',
        content: `I've uploaded: ${fileNames}`
      };

      const botResponse: ChatMessage = {
        type: 'assistant',
        content: `I've received your file${files.length > 1 ? 's' : ''}: ${fileNames}. How can I help you with this?`,
      };

      setState(prev => ({
        ...prev,
        chats: prev.chats.map(chat =>
          chat.id === activeChat
            ? { ...chat, messages: [...chat.messages, newUserMessage, botResponse] }
            : chat
        ),
      }));

      setShowFileUpload(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white shadow-sm py-4 px-6">
        <div className="max-w-4xl mx-auto flex justify-between items-center">
          <div className="flex items-center">
            <div className="bg-blue-500 rounded-full p-2 mr-3">
              <Bot size={20} className="text-white" />
            </div>
            <h1 className="font-semibold text-lg text-gray-800">FlexAI</h1>
          </div>
          <div className="flex items-center">
            <div className="bg-teal-600 rounded-full p-2 mr-2">
              <User size={18} className="text-white" />
            </div>
            <span className="text-sm font-medium text-gray-700">{userInfo.name}</span>
          </div>
        </div>
      </header>

      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6">
        <div className="max-w-3xl mx-auto space-y-4">
          {activeMessages.map((message, idx) => (
            <ChatBubble key={idx} message={message} />
          ))}
          {isTyping && (
            <div className="flex items-start max-w-[80%] md:max-w-[70%]">
              <div className="bg-white rounded-2xl rounded-tl-none py-3 px-4 shadow-sm">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {showFileUpload && (
        <FileUpload
          onClose={() => setShowFileUpload(false)}
          onUpload={handleFileUpload}
          fileInputRef={fileInputRef}
        />
      )}

      {/* Input area */}
      <div className="bg-white border-t border-gray-200 p-4">
        {!sessionReady && (
          <div className="text-sm text-gray-500 text-center pb-2">Initializing session...</div>
        )}
        <div className="max-w-3xl mx-auto flex items-end">
          <button
            onClick={() => setShowFileUpload(true)}
            disabled={!sessionReady}
            className="p-3 rounded-full hover:bg-gray-100 transition-colors mr-2 text-gray-500 hover:text-blue-500"
            aria-label="Upload file"
          >
            <FileUp size={20} />
          </button>

          <div className="flex-1 relative">
            <input
              type="text"
              value={input}
              disabled={!sessionReady}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message here..."
              className="w-full rounded-full py-3 px-4 pr-12 border border-gray-300 disabled:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={handleSendMessage}
              disabled={!input.trim() || !sessionReady}
              className={`absolute right-3 top-1/2 transform -translate-y-1/2 p-1.5 rounded-full ${
                input.trim() && sessionReady
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 text-gray-400'
              } focus:outline-none`}
              aria-label="Send message"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface; 