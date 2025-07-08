import { useState, useEffect } from 'react';
import UserInfoForm from './components/UserInfoForm';
import ChatInterface from './components/ChatInterface';
import { AppState, UserInfo, Chat, ChatMessage } from './types';

// Add this declaration to fix the import.meta.env error
interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  // add other env variables here if needed
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

function App() {
  const [state, setState] = useState<AppState>({
    step: 'user-info',
    userInfo: { name: '', email: '', phone: '' },
    chats: [],
    activeChat: null,
    isTyping: false,
    sessionId: null,
    userId: null,
  });
 
  useEffect(() => {
    const initializeSession = async () => {
      const storedSessionId = localStorage.getItem('sessionId');
      const storedUserId = localStorage.getItem('userId');

      if (storedSessionId && storedUserId) {
        setState(prev => ({
          ...prev,
          sessionId: storedSessionId,
          userId: storedUserId,
        }));
      }
    };

    initializeSession();
  }, []);

  // Updated to match ChatMessage interface
  const createWelcomeMessage = (name: string): ChatMessage => ({
    type: 'assistant',
    content: `Hi ${name}, this is FlexAI - your workspace booking assistant! What do you need help with today?`,
    additional_kwargs: {},
    response_metadata: {},
    tool_calls: [],
    invalid_tool_calls: [],
    // attachments is optional, omit unless needed
  });

  const handleUserInfoSubmit = async (userInfo: UserInfo) => {
    const API_BASE = import.meta.env.VITE_API_URL;
    try {
      const res = await fetch(`${API_BASE}/session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_info: userInfo }),
      });

      const data = await res.json();

      if (data.session_id && data.user_id) {
        localStorage.setItem('sessionId', data.session_id);
        localStorage.setItem('userId', data.user_id);
        console.log('Session created in App.tsx:', data.session_id);
        const initialChat: Chat = {
          id: Date.now().toString(),
          title: 'New Chat',
          messages: [createWelcomeMessage(userInfo.name)],
        };

        setState(prev => ({
          ...prev,
          step: 'chat',
          userInfo,
          sessionId: data.session_id,
          userId: data.user_id,
          chats: [initialChat],
          activeChat: initialChat.id,
        }));
      } else {
        console.error('Invalid session response');
      }
    } catch (err) {
      console.error('Failed to create session:', err);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      {state.step === 'user-info' ? (
        <UserInfoForm onSubmit={handleUserInfoSubmit} />
      ) : state.sessionId && state.userId ? (
        <ChatInterface
          userInfo={state.userInfo}
          chats={state.chats}
          activeChat={state.activeChat}
          isTyping={state.isTyping}
          setState={setState}
          sessionId={state.sessionId}
          user_Id={state.userId}
        />
      ) : (
        <div className="flex items-center justify-center h-screen text-gray-600 text-lg">
          Initializing session...
        </div>
      )}
    </div>
  );
}

export default App;