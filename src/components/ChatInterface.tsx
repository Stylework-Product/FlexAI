import React, { useState, useRef, useEffect } from 'react';
import { AppState, ChatMessage, UserInfo, Chat } from '../types';
import { Send, FileUp, User, Bot, Upload, X, File } from 'lucide-react';
import ChatBubble from './ChatBubble';
 
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
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [documentUploaded, setDocumentUploaded] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const sessionReady = Boolean(sessionId && user_Id);
  const activeMessages = chats.find(chat => chat.id === activeChat)?.messages || [];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeMessages]);

  useEffect(() => {
    // Try to load existing document when component mounts
    if (sessionReady) {
      loadExistingDocument();
    }
  }, [sessionReady]);

  const loadExistingDocument = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/load_document', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });

      const data = await response.json();
      if (response.ok && !data.error) {
        setDocumentUploaded(true);
        console.log('Document loaded successfully');
      }
    } catch (error) {
      console.log('No existing document found');
    }
  };

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
      const response = await fetch('http://127.0.0.1:8000/gemini_chat', {
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

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      setSelectedFile(file);
    } else {
      alert('Please select a PDF file');
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile || !sessionId) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('session_id', sessionId);

    try {
      const response = await fetch('http://127.0.0.1:8000/upload_document', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setDocumentUploaded(true);
        setShowFileUpload(false);
        setSelectedFile(null);

        // Add success message to chat
        const successMessage: ChatMessage = {
          type: 'assistant',
          content: `✅ Document "${data.filename}" uploaded successfully! You can now ask questions about the document content.`,
          timestamp: new Date(),
        };

        setState(prev => ({
          ...prev,
          chats: prev.chats.map(chat =>
            chat.id === activeChat
              ? { ...chat, messages: [...chat.messages, successMessage] }
              : chat
          ),
        }));
      } else {
        throw new Error(data.detail || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert('Failed to upload document. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const FileUploadModal = () => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl max-w-md w-full shadow-2xl">
        <div className="p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-800">Upload PDF Document</h3>
            <button
              onClick={() => {
                setShowFileUpload(false);
                setSelectedFile(null);
              }}
              className="text-gray-400 hover:text-gray-600 focus:outline-none transition-colors"
            >
              <X size={20} />
            </button>
          </div>
          
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center mb-4">
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-2" />
            <p className="text-sm text-gray-600 mb-2">
              Select a PDF document to upload
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={handleFileSelect}
            />
            <button
              type="button"
              className="text-blue-500 hover:text-blue-600 font-medium focus:outline-none focus:underline transition-colors"
              onClick={() => fileInputRef.current?.click()}
            >
              Choose PDF File
            </button>
          </div>
          
          {selectedFile && (
            <div className="mb-4 p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center">
                <File size={16} className="text-blue-500 mr-2" />
                <span className="text-sm text-gray-700 truncate">
                  {selectedFile.name}
                </span>
              </div>
            </div>
          )}
          
          <div className="flex justify-end space-x-3">
            <button
              type="button"
              onClick={() => {
                setShowFileUpload(false);
                setSelectedFile(null);
              }}
              className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 text-sm font-medium hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleFileUpload}
              disabled={!selectedFile || isUploading}
              className={`px-4 py-2 rounded-lg text-white text-sm font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors ${
                !selectedFile || isUploading
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-blue-500 hover:bg-blue-600'
              }`}
            >
              {isUploading ? 'Uploading...' : 'Upload'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white shadow-sm py-4 px-6">
        <div className="max-w-4xl mx-auto flex justify-between items-center">
          <div className="flex items-center">
            <div className="bg-blue-500 rounded-full p-2 mr-3">
              <Bot size={20} className="text-white" />
            </div>
            <div>
              <h1 className="font-semibold text-lg text-gray-800">FlexAI</h1>
              <p className="text-xs text-gray-500">
                Workspace recommendations & document Q&A
                {documentUploaded && (
                  <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                    Document Ready
                  </span>
                )}
              </p>
            </div>
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

      {showFileUpload && <FileUploadModal />}

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
            aria-label="Upload PDF document"
            title="Upload PDF document for Q&A"
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
              placeholder={documentUploaded 
                ? "Ask about workspaces or your uploaded document..." 
                : "Ask about workspaces or upload a document for Q&A..."
              }
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