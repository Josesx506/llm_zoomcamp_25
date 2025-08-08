'use client';

import HistoryBtn from '@/components/HistoryBtn';
import { API_BASE_URL } from '@/lib/config';
import styles from '@/styles/history.module.css';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import toast from 'react-hot-toast';
import { PiChatTeardropTextBold } from "react-icons/pi";

export default function History() {
  const [loading, setLoading] = useState(false);
  const [convos,setConvos] = useState([]);

  useEffect(() => { // Retrieve prior conversations
    const controller = new AbortController();
    const signal = controller.signal;

    async function fetchConversations() {
      try {
        setLoading(true);
        const res = await fetch(`${API_BASE_URL}/conversations`, { signal: signal });

        if (!res.ok) {
          const errorData = await res.json();
          throw new Error(errorData.detail || "Failed to fetch conversations");
        }

        const data = await res.json();
        setConvos(data.conversations);
      } catch (err) {
        if (err?.code !== "ERR_CANCELED" && err.name !== "AbortError") {
          toast.error(err.message);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchConversations();
    return () => { controller.abort() };
  }, []);

  function deleteConversation(conv_id) {
    const remaining = convos.filter((conv)=>(conv.conv_id!=conv_id))
    setConvos(remaining);
  }

  return (
    <aside className={styles.chatHistory}>
      <div className={styles.newChat}>
        <Link className={styles.newChatBtn} href={"/chat"}>
          <PiChatTeardropTextBold size={"1rem"} /> <span>New Chat</span>
        </Link>
      </div>
      <div className={styles.prevChats}>
        <h3>Chats</h3>
        <div className={styles.prevChatList}>
          {convos.map((conv)=>(<HistoryBtn key={conv.conv_id} updateConvoList={deleteConversation} {...conv} />))}
        </div>
      </div>
    </aside>
  )
}
