import ChatPage from "@/components/pages/ChatPage";

export default async function page({ params }) {
  const { conv_id } = await params;
  return (
    <ChatPage conv_id={conv_id} />
  )
}
