import cv2
import random
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt


class AugmentImageDataSet:

    def __init__(self, dataset_source: Path, destination: Path):
        # .resolve() converts relative paths to absolute, so all operations
        # are anchored to the filesystem regardless of the notebook's CWD.
        self.source      = Path(dataset_source).resolve()
        self.destination = Path(destination).resolve()

        self._original_df: pd.DataFrame  = None
        self._augmented_df: pd.DataFrame = None

        os.makedirs(str(self.destination), exist_ok=True)

        for folder in self.source.rglob("*"):
            if any(part.startswith(".") for part in folder.parts):
                continue
            if folder.is_dir():
                dest_path = os.path.join(
                    str(self.destination),
                    str(folder.relative_to(self.source))
                )
                os.makedirs(dest_path, exist_ok=True)

        self._original_df = self._scan_folder(self.source, "Original")
        # 0 — original image, not used as augmentation source
        self._original_df["augmented"] = 0


    # ─────────────────────────────────────────────────────────────────────────
    def _scan_folder(self, folder_path: Path, source_label: str) -> pd.DataFrame:
        rows = []
        for class_folder in sorted(folder_path.iterdir()):
            if not class_folder.is_dir() or class_folder.name.startswith("."):
                continue
            if any(part.startswith(".") for part in class_folder.relative_to(folder_path).parts):
                continue
            for img in sorted(class_folder.glob("*.jpg")):
                if any(part.startswith(".") for part in img.relative_to(folder_path).parts):
                    continue
                rows.append({
                    "Filepath": str(img),
                    "Label"   : class_folder.name,
                    "Source"  : source_label,
                })
        # Always return a DataFrame with the expected schema, even when no images are found.
        # Without this, an empty rows list produces a column-less DataFrame, which causes
        # KeyError crashes in validate() and get_df().
        cols = ["Filepath", "Label", "Source"]
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


    # ─────────────────────────────────────────────────────────────────────────
    def augment_by_rotation(self, target_per_class: int = 3500, max_angle: float = 60):
        # Reset all originals to 0 — not yet used as augmentation source
        self._original_df["augmented"] = 0

        # Clear existing augmented images to avoid duplicates on re-runs
        for class_folder in sorted(self.destination.iterdir()):
            if not class_folder.is_dir() or class_folder.name.startswith("."):
                continue
            for img_file in class_folder.glob("*.jpg"):
                img_file.unlink()

        for label, class_rows in self._original_df.groupby("Label"):
            current_count = len(class_rows)
            needed        = target_per_class - current_count

            if needed <= 0:
                print(f"  {label:<20} {current_count} images — no augmentation needed")
                continue

            print(f"  {label:<20} {current_count} images — generating {needed}")

            dest_class = self.destination / label
            # Guarantee the class folder exists before writing — the __init__ mirror
            # may have run with a different CWD or the folder may have been deleted.
            dest_class.mkdir(parents=True, exist_ok=True)
            generated  = 0

            while generated < needed:
                row             = class_rows.sample(1).iloc[0]
                orig_idx        = row.name
                source_img_path = Path(row["Filepath"])

                image = cv2.imread(str(source_img_path))
                if image is None:
                    continue

                # 1 — original image that has been used as augmentation source
                self._original_df.at[orig_idx, "augmented"] = 1

                angle           = random.uniform(-max_angle, max_angle)
                height, width   = image.shape[:2]
                center          = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated         = cv2.warpAffine(
                    image, rotation_matrix, (width, height),
                    borderMode=cv2.BORDER_REPLICATE
                )

                save_path = dest_class / f"{source_img_path.stem}_aug_{generated}.jpg"
                ok = cv2.imwrite(str(save_path), rotated)
                if not ok:
                    raise RuntimeError(
                        f"cv2.imwrite failed — check path/permissions: {save_path}"
                    )
                generated += 1

        # Rebuild augmented dataframe and mark all rows as 2 — copied/augmented images
        self._augmented_df = self._scan_folder(self.destination, "Augmented")
        if self._augmented_df.empty:
            import warnings
            warnings.warn(
                "augment_by_rotation() finished but the destination scan found no images. "
                f"Check that files were written to: {self.destination}"
            )
        self._augmented_df["augmented"] = 2


    # ─────────────────────────────────────────────────────────────────────────
    def get_df(self, source: str = "Original") -> pd.DataFrame:
        assert source in ("Original", "Augmented"), "source must be 'Original' or 'Augmented'"

        if source == "Original":
            return self._original_df
        else:
            # _augmented_df may have been set by augment_by_rotation() but could still be
            # empty (e.g. images failed to write or the scan found nothing). Always recheck.
            if self._augmented_df is not None and not self._augmented_df.empty:
                return self._augmented_df
            # Either never set, or was set to an empty frame — (re-)scan from disk.
            self._augmented_df = self._scan_folder(self.destination, "Augmented")
            if self._augmented_df.empty:
                raise ValueError(
                    "Augmented dataset is empty. "
                    "Please run augment_by_rotation() first."
                )
            # Assign 2 — augmented copies loaded from disk
            self._augmented_df["augmented"] = 2
            return self._augmented_df


    # ─────────────────────────────────────────────────────────────────────────
    def validate(self, source):
        if isinstance(source, str):
            if source in ("Original", "Augmented"):
                df = self.get_df(source)
            else:
                raise TypeError("source must be 'Original', 'Augmented', or a DataFrame")
        elif isinstance(source, pd.DataFrame):
            df = source
        else:
            raise TypeError("source must be 'Original', 'Augmented', or a DataFrame")

        # Guard required columns regardless of how df was obtained.
        required_cols = {"Filepath", "Label", "Source"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        print("━" * 50)
        print("DATAFRAME INFO")
        print("━" * 50)
        print(df.info())
        print()

        nulls = df.isnull().sum()
        print("Null values per column:")
        print(nulls)
        print()

        n_dupes = df["Filepath"].duplicated().sum()
        print(f"Duplicate filepaths: {n_dupes}")
        if n_dupes > 0:
            print("Duplicate rows preview:")
            print(df[df["Filepath"].duplicated()].head())
        print()

        print("Class distribution:")
        print(df["Label"].value_counts())
        print()

        print(f"Total samples  : {len(df)}")
        print(f"Unique classes : {df['Label'].nunique()}")

        if "augmented" in df.columns:
            n_untouched = int((df["augmented"] == 0).sum())
            n_source    = int((df["augmented"] == 1).sum())
            n_copies    = int((df["augmented"] == 2).sum())
            print(f"\n  augmented = 0  (original, unused)     : {n_untouched}")
            print(f"  augmented = 1  (original, used source) : {n_source}")
            print(f"  augmented = 2  (augmented copy)        : {n_copies}")
        print("━" * 50)


    # ─────────────────────────────────────────────────────────────────────────
    def show_samples(self, random_state=42):
        orig_df = self.get_df("Original")
        try:
            aug_df = self.get_df("Augmented")
        except ValueError as e:
            print(f"Error: {e}")
            return

        orig_labels = orig_df["Label"].unique()
        aug_labels  = aug_df["Label"].unique()

        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        fig.suptitle(
            "Sample Images: Original (Top 2 Rows) vs Augmented (Bottom 2 Rows)",
            fontsize=14, fontweight="bold"
        )
        axes = axes.flatten()
        idx  = 0

        for label in orig_labels:
            sample = orig_df[orig_df["Label"] == label].sample(1, random_state=random_state)
            img    = cv2.imread(sample.iloc[0]["Filepath"])
            if img is not None:
                axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(f"{label}\n[Original]", fontsize=10)
            axes[idx].axis("off")
            idx += 1

        for label in aug_labels:
            sample = aug_df[aug_df["Label"] == label].sample(1, random_state=random_state + 1)
            img    = cv2.imread(sample.iloc[0]["Filepath"])
            if img is not None:
                axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(f"{label}\n[Augmented]", fontsize=10)
            axes[idx].axis("off")
            idx += 1

        for ax in axes[idx:]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()


    # ─────────────────────────────────────────────────────────────────────────
    def plot_distribution(self):
        orig_df = self.get_df("Original")
        aug_df  = self.get_df("Augmented")
        dist    = pd.concat([orig_df, aug_df], ignore_index=True)

        summary = dist.groupby(["Label", "Source"]).size().unstack(fill_value=0)
        summary["Total"] = summary.sum(axis=1)
        summary = summary.sort_values("Total", ascending=False)

        print("Image count per cell type & source:")
        print(summary.to_string())
        print(f"\nGrand total: {summary['Total'].sum():,}\n")

        plt.style.use("dark_background")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("WBC Dataset — Image Distribution", fontsize=13, color="#00e5ff")

        ACCENT = "#00e5ff"
        WARN   = "#ff6b6b"

        plot_df = summary.drop(columns="Total")
        if "Original" in plot_df.columns and "Augmented" in plot_df.columns:
            plot_df = plot_df[["Original", "Augmented"]]

        plot_df.plot(kind="bar", ax=axes[0], color=[ACCENT, WARN], edgecolor="#222", width=0.7)
        axes[0].set_title("Images per Class (Original vs Augmented)", color="#ccc")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=35)
        axes[0].legend(facecolor="#1a1a1a", edgecolor="#444")
        axes[0].grid(axis="y", color="#333", linestyle="--", alpha=0.5)

        for bar in axes[0].patches:
            h = bar.get_height()
            if h > 0:
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2, h + 20,
                    f"{int(h):,}", ha="center", va="bottom", fontsize=7, color="#aaa"
                )

        source_counts = dist["Source"].value_counts()
        axes[1].pie(
            source_counts,
            labels=source_counts.index,
            autopct="%1.1f%%",
            colors=[ACCENT, WARN],
            startangle=90,
            textprops={"color": "#eee"},
            wedgeprops={"edgecolor": "#333"},
        )
        axes[1].set_title("Original vs Augmented Split", color="#ccc")

        plt.tight_layout()
        plt.show()


    # ─────────────────────────────────────────────────────────────────────────
    def get_dataframe(self) -> pd.DataFrame:
        return pd.concat(
            [self.get_df("Original"), self.get_df("Augmented")],
            ignore_index=True
        )