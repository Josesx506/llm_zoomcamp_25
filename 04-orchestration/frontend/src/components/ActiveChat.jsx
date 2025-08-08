'use client';

import ChatMessage from '@/components/ChatMessage';
import { API_BASE_URL } from '@/lib/config';
import styles from '@/styles/activechat.module.css';
import Form from 'next/form';
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import TextareaAutosize from 'react-textarea-autosize';

export default function ActiveChat({ conv_id }) {
  const [conversationId, setConversationId] = useState(conv_id);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const textInputRef = useRef();
  const newMsgRef = useRef();

  useEffect(() => { // Retrieve existing messages
    if (!conv_id) return;
    const controller = new AbortController();
    const signal = controller.signal;

    async function fetchMessages() {
      try {
        setLoading(true);
        const res = await fetch(`${API_BASE_URL}/conversations/${conv_id}/messages`, {
          signal: signal,
        });

        if (!res.ok) {
          const errorData = await res.json();
          throw new Error(errorData.detail || "Failed to fetch messages");
        }

        const data = await res.json();
        setMessages(data.messages);
      } catch (err) {
        if (err?.code !== "ERR_CANCELED" && err.name !== "AbortError") {
          toast.error(err.message || "Failed to load conversation history");
        }
      } finally {
        setLoading(false);
      }
    };

    fetchMessages();
    return () => { controller.abort() };
  }, [conv_id]);

  useEffect(() => {
    // Scroll to the bottom of the chats
    if (newMsgRef.current) {
      newMsgRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  async function submitMessage(e) {
    e.preventDefault();
    let query = textInputRef.current.value;

    try {
      setLoading(true);
      const payload = { prompt: query };
      if (conversationId) { // include conv_id for existing conversations
        payload.conversation_id = conversationId;
      }
      const res = await fetch(`${API_BASE_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) { // Catch non-2xx status code and throw errors
        const errorData = await res.json();
        throw new Error(errorData.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();
      data["query"] = query;

      // Update conversation ID if starting a new conversation
      if (!conversationId && data.conversation_id) {
        setConversationId(data.conversation_id);
        router.push(`/chat/${data.conversation_id}`)
      }

      setMessages((prev) => [...prev, data])

    } catch (err) {
      if (err?.code !== "ERR_CANCELED") {
        toast.error(err.message || 'Fetch error');
      }
    } finally {
      setLoading(false);
      textInputRef.current.value = ""; // Reset form text
    }
  }

  return (
    <div className={styles.activeConversation}>
      <div className={styles.chatResults}>
        {messages.map((msg) => (<ChatMessage key={msg.msg_id} conv_id={conv_id} {...msg} />))}
        <div id='newmsg' ref={newMsgRef} />
      </div>
      <div className={styles.lowerRow}>
        <Form className={styles.chatForm} onSubmit={submitMessage}>
          <div>
            <label htmlFor="chattext">
              <TextareaAutosize id="chattext" className={styles.chattext}
                placeholder="Enter chat ...." name="chattext" minRows={3}
                maxRows={7} wrap="soft" ref={textInputRef} required />
            </label>
          </div>
          <div className={styles.chatSubmit}>
            <button disabled={loading}>Submit</button>
          </div>
        </Form>
        <div className={styles.disclaimer}>LLMs can make mistakes. Check important info.</div>
      </div>
    </div>
  )
}
