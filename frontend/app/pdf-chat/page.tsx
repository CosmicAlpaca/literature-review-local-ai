import AppShell from "@/components/app/AppShell";
import PdfChatPage from "@/components/pdf-chat/PdfChatPage";

export const metadata = {
  title: "Chat với PDF | Literature Review",
  description: "Tải lên tài liệu PDF và đặt câu hỏi bằng AI.",
};

export default function Page() {
  return (
    <AppShell>
      <div className="h-full">
        <PdfChatPage />
      </div>
    </AppShell>
  );
}
