'use client';

import { API_BASE_URL } from '@/lib/config';
import styles from '@/styles/historybtn.module.css';
import Link from 'next/link';
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { IoEllipsisHorizontal } from "react-icons/io5";
import { RiDeleteBinLine } from "react-icons/ri";

export default function HistoryBtn({ conv_id, title, updateConvoList=()=>{} }) {
  const pathname = usePathname();
  const router = useRouter();
  const buttonRef = useRef(null);
  const popupRef = useRef(null);
  const [visiblePopup, setVisiblePopup] = useState(false);
  const [popupPos, setPopupPos] = useState({ top: 0, left: 0 });

  const togglePopup = () => {
    if (buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      setPopupPos({ top: rect.bottom + 6, left: rect.left });
    }
    setVisiblePopup((prev) => !prev);
  };

  // Close the popup if clicked outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (buttonRef.current && !buttonRef.current.contains(event.target) &&
        popupRef.current && !popupRef.current.contains(event.target)) {
        setVisiblePopup(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [buttonRef, popupRef]);

  async function deleteConversation(e) {
    try {
      const res = await fetch(`${API_BASE_URL}/conversations/${conv_id}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" }
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Delete conversation error");
      }
      toast.success("Conversation deleted")
    } catch (err) {
      if (err?.code !== "ERR_CANCELED" && err.name !== "AbortError") {
        toast.error(err.message);
      }
    } finally {
      setVisiblePopup(false);
      updateConvoList(conv_id); // Remove the deleted conversation from the history
      if (pathname.endsWith(conv_id)) {
        router.push("/chat");
      }
    }
  }

  return (
    <div className={`${styles.histLink} ${pathname.endsWith(conv_id) ? styles.active : ''}`}>
      <Link href={`/chat/${conv_id}`}>
        {title}
      </Link>
      <IoEllipsisHorizontal className={styles.modal} ref={buttonRef} onClick={togglePopup} />
      {visiblePopup && (
        <div className={styles.popup} ref={popupRef} style={{ top: popupPos.top, left: popupPos.left, position: "fixed" }}>
          <div className={styles.deletebtn} onClick={deleteConversation}>
            <RiDeleteBinLine /> Delete
          </div>
        </div>
      )}
    </div>
  )
}
