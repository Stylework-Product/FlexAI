export interface UserInfo {
  name: string;
  email: string;
  phone: string;
}

export interface AppState {
  step: 'user-info' | 'chat';
  userInfo: UserInfo;
  chats: Chat[];
  activeChat: string | null;
  isTyping: boolean;
  sessionId: string | null;
  userId: string | null;
}

export interface ChatMessage {
  type: 'user' | 'assistant';
  content: string;
  additional_kwargs?: Record<string, any>;
  response_metadata?: Record<string, any>;
  tool_calls?: any[]; 
  invalid_tool_calls?: any[]; 
  timestamp?: Date;
  attachments?: { name: string; [key: string]: any }[];
  recommendations?: any[];
}

export interface Chat {
  id: string;
  title: string;
  messages: ChatMessage[];
}

export interface AppState {
  step: 'user-info' | 'chat';
  userInfo: UserInfo;
  chats: Chat[];
  activeChat: string | null;
  isTyping: boolean;
  sessionId: string | null;
}