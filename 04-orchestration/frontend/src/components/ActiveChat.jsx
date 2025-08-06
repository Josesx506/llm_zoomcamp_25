'use client';

import ChatMessage from '@/components/ChatMessage';
import styles from '@/styles/activechat.module.css';
import Form from 'next/form';
import { useEffect, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import TextareaAutosize from 'react-textarea-autosize';

export default function ActiveChat() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const textInputRef = useRef();
  const newMsgRef = useRef();

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
      const response = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: query })
      });
      
      if (!response.ok) { // Catch non-2xx status code and throw errors
        const errorData = await response.json();
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      data["query"] = query;
      setMessages((prev) => [...prev, data])

    } catch (err) {
      if (err?.code !== "ERR_CANCELED") {
        toast.error(err.message || 'Fetch error');
      }
    } finally {
      setLoading(false);
      // Reset form text
      textInputRef.current.value = "";
    }
  }

  return (
    <div className={styles.activeConversation}>
      <div className={styles.chatResults}>
        {messages.map((msg) => (<ChatMessage key={msg.id} {...msg} />))}
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
