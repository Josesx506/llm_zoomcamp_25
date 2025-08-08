'use client';

import styles from '@/styles/chatmsg.module.css';
import { API_BASE_URL } from '@/lib/config';
import { AiOutlineDislike, AiOutlineLike } from "react-icons/ai";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useState } from 'react';
import toast from 'react-hot-toast';
import Tooltip from 'rc-tooltip';
import 'rc-tooltip/assets/bootstrap_white.css';

export default function ChatMessage({ conv_id, msg_id, query, response, like_dislike }) {
  const [loading, setLoading] = useState(false);
  const [vote, setVote] = useState(parseInt(like_dislike))

  async function upvote(e) {
    e.preventDefault()
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE_URL}/conversations/${conv_id}/${msg_id}/upvote`)
      if (!res.ok) { // Catch non-2xx status code and throw errors
        const errorData = await res.json();
        throw new Error(errorData.detail || `Server error: ${res.status}`);
      }
      const data = await res.json();
      setVote(data.vote);
    } catch (err) {
      if (err?.code !== "ERR_CANCELED" && err.name !== "AbortError") {
        toast.error(err.message || "Failed to upvote response");
      }
    } finally {
      setLoading(false);
    }
  }

  async function downvote(e) {
    e.preventDefault()
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE_URL}/conversations/${conv_id}/${msg_id}/downvote`)
      if (!res.ok) { // Catch non-2xx status code and throw errors
        const errorData = await res.json();
        throw new Error(errorData.detail || `Server error: ${res.status}`);
      }
      const data = await res.json();
      setVote(data.vote);
    } catch (err) {
      if (err?.code !== "ERR_CANCELED" && err.name !== "AbortError") {
        toast.error(err.message || "Failed to upvote response");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.msgCntr}>
      <div className={styles.query}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{query}</ReactMarkdown>
      </div>
      <div className={styles.response}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{response}</ReactMarkdown>
      </div>
      <div className={styles.review}>
        <Tooltip overlay={<span>{vote == -1 ? "Neutral" : "Upvote"}</span>} placement="right" classNames={{body: styles.customTooltip}}>
          <button style={{ backgroundColor: vote == 1 ? "lightgreen" : "white" }} onClick={upvote} disabled={loading} ref={null}><AiOutlineLike /></button>
        </Tooltip>
        <Tooltip overlay={<span>{vote == 1 ? "Neutral" : "Downvote"}</span>} placement="left" classNames={{body: styles.customTooltip}}>
          <button style={{ backgroundColor: vote == -1 ? "rgba(204,72,82,0.5)" : "white" }} onClick={downvote} disabled={loading} ref={null}><AiOutlineDislike /></button>
        </Tooltip>
      </div>
    </div>
  )
}
