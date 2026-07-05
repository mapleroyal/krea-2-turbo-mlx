export const APP_NAME = "Krea 2 Turbo";
export const APP_VERSION = import.meta.env.VITE_APP_VERSION ?? "0.1.0";
export const APP_DESCRIPTION =
  "Local Krea 2 Turbo generation workspace for krea-2-turbo-mlx.";
export const APP_STORAGE_PREFIX = "krea-2-turbo-mlx";

export const THEME_MODES = ["system", "light", "dark"];
export const THEME_STORAGE_KEY = `${APP_STORAGE_PREFIX}:theme`;

export function formatPageTitle(title) {
  return title ? `${title} - ${APP_NAME}` : APP_NAME;
}
