import { createBrowserRouter } from "react-router-dom";
import PageContainer from "@/components/layout/PageContainer";
import ExecutiveDashboard from "@/pages/ExecutiveDashboard";
import CustomerRisk from "@/pages/CustomerRisk";
import RetentionRecommendations from "@/pages/RetentionRecommendations";
import AIRetentionAssistant from "@/pages/AIRetentionAssistant";
import ModelPerformance from "@/pages/ModelPerformance";
import NotFound from "@/pages/NotFound";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <PageContainer />,
    children: [
      { index: true, element: <ExecutiveDashboard /> },
      { path: "customer-risk", element: <CustomerRisk /> },
      { path: "recommendations", element: <RetentionRecommendations /> },
      { path: "assistant", element: <AIRetentionAssistant /> },
      { path: "model-performance", element: <ModelPerformance /> },
      { path: "*", element: <NotFound /> },
    ],
  },
]);
