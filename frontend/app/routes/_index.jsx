import { KreaWorkspace } from "@/components/krea-workspace";
import { APP_DESCRIPTION, formatPageTitle } from "@/lib/app-config";

export function meta() {
  return [
    { title: formatPageTitle("Workspace") },
    { name: "description", content: APP_DESCRIPTION },
  ];
}

export default function HomeRoute() {
  return <KreaWorkspace />;
}
