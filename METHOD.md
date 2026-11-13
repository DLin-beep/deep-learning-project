# Method

Every experiment follows the same two steps. A network pretrained on ImageNet first keeps training on a source, such
as PlantVillage leaves. We then freeze it and fit a small classifier on a few labeled images of a medical target, such
as HAM10000 skin lesions. Running the same test on the network without the extra training tells us what the source
added. We call the difference in balanced accuracy the transfer gain, measured in percentage points and averaged over
the number of labels. A positive gain means the source helped. This file covers how we ran the study, and the paper
reports the results.

## Sources and source training

The main screen compares four sources. PlantVillage has photos of healthy and diseased leaves, and we use its segmented
version so the network sees the leaf and its lesions rather than the background. NCT-CRC-HE-100K has histopathology
tiles. Plants and histopathology are both disease domains, one outside medicine and one inside it. DTD has many of the
same dotted and blotchy patterns, but no disease. We leave out its freckled and veined classes, which mostly show skin
or leaves. The fourth source, 38 classes of Food-101, gives close-up, fine-grained photos with no disease at all.

Every source is cut down to 5,400 images, the size of DTD. Each one trains for 840 steps of 64 images with SGD
(learning rate 0.01, cosine schedule) and the same augmentation, with three seeds. Only the images and labels differ
between sources. We also train on grayscale, blurred and tile-shuffled versions of PlantVillage. These take away the
color, the fine dots and the overall shape of the leaf, so we can see which of these properties matter.

## Backbones and readouts

Every condition runs on four backbones, all pretrained on the same ImageNet-1k images. Two are supervised, a ResNet-50
and a ViT-S/16, and the other two are their self-supervised DINO versions. ImageNet-trained CNNs lean heavily on
texture. Comparing the ResNet with the ViT, and supervised with self-supervised pretraining, helps us separate the
effect of the source from that of the model. Source training updates stages 3 and 4 of the ResNet-50, or the last 6 of
the 12 ViT blocks, along with a new classification head.

For the few-shot tests, we read out pooled features after stages 2, 3 and 4 (or blocks 6, 9 and 12), their
concatenation, and, for the ViTs, the class token. Each condition's readout is chosen on a validation split. We compare
it with the ImageNet network at that network's own best readout, because a single fixed readout can make transfer
appear where there is none.

## Targets and few-shot evaluation

The screen has three targets. HAM10000 has skin lesions, DDR has retina photos graded for diabetic retinopathy on five
levels, and ChestX-ray14 has chest X-rays, of which we use BSCD-FSL's seven-class subset, capped at 12,000 images.

For each condition, the features stay frozen and a multinomial logistic regression with an L2 penalty learns from k
labeled images per class, with k set to 5, 20 or 50. Its regularization strength is fixed in advance at C = 0.1, since
there are too few examples to tune it at small k. Few-shot results depend a lot on which images get drawn, so we draw
the labeled images 20 times for each k. Every condition and backbone gets the same draws. Related images, such as
several photos of one lesion or scans of one patient, always stay in the same split. We use five such splits, each
with its own test set. As a reference, we also fit a classifier on all the labeled training images. We also fine-tune
the whole network on 3 of the 20 draws for k of 20 and 50, starting from the logistic regression.

Most HAM10000 lesions had already been used in pilot experiments. After the screen, the official ISIC 2018 Task 3 test
set became our held-out test. It has 1,512 images of 1,223 lesions, and none of them are among HAM10000's 7,470. We
score it with the classifiers fitted on HAM10000 shots, using six seeds per source for DINO ViT-S, and we never used it
to fit a classifier or choose a setting.

## Metrics and statistics

About two-thirds of HAM10000's images are nevi, so our main metric is balanced accuracy, the average recall over
classes. The 95% confidence intervals come from a hierarchical bootstrap over seeds, splits and test images. Test
images are grouped by lesion or patient where we know them. All splits share the ISIC 2018 test set, so each bootstrap
draw resamples its 1,223 lesions once for all of them. A gain counts as significant when its interval excludes zero.
The intervals are not corrected for multiple comparisons, except where a result is marked as Holm-corrected.

## Attribute distance

Our measure of shared visual properties does not depend on the networks we compare. For 1,500 images of each dataset,
we compute simple hand-crafted features. They count blobs at several sizes (dots and specks), measure line and edge
filter responses (streaks and networks), and add texture histograms and color statistics. Each feature is standardized
over the screen's ten datasets, its seven sources and three targets. The attribute distance between two datasets is
then the Fréchet distance between their feature distributions. We compute it with and without the color statistics,
since chest X-rays are grayscale and would sit far from every source on color alone.

For overall image similarity, we use the Fréchet distance between the datasets' ImageNet features, as Matsoukas et al.
did. For each backbone, we regress the gains of its 12 source-target pairs on both distances, with a separate
intercept for each target. A permutation test checks the attribute coefficient by shuffling the attribute distances
among the sources within each target. If shared properties mattered, DTD would help as much as plants or
histopathology at the same distance, and chest X-rays would gain the least.

## What carries the gain

Taking properties away from the leaves is one test. Changing the task while keeping the same images is another. The
same PlantVillage images get labeled by species only, by the original 38 species-and-disease classes, or as healthy
versus diseased. Other objects get the same treatment, such as VisA's manufactured products, labeled by defect or by
object type.

Not only do we compare tasks, but we also look inside the networks. Linear probes on the frozen features predict the
expert-labeled skin attributes of ISIC 2018 Task 2, such as globules and streaks, which correspond to plant cues such
as dark specks and irregular margins. For DINO ViT-S, LEACE erases these attribute directions, and we compare the drop
in the gain with the drop from erasing the same number of random directions. We also crop sources and targets to their
objects, since how much of the frame an object fills could matter as much as what it shows.

Synthetic lesions test the task directly. Each one is a smoothed random polygon covering 1 to 20% of the object. It is
Poisson-blended in from another image and shifted in color. We paste these lesions onto healthy PlantVillage leaves
and COIL-100 objects. DINO ViT-S then trains with the same recipe to score each lesioned image above its clean copy,
and we call the result a synthetic-pair start.

A last question is whether plant training adds new features or just reweights the ones the network already has. We fit
a linear map from the ImageNet network's features to the plant-trained network's, on HAM10000 and without labels.
ImageNet's ISIC features then pass through this map, and we measure how much of the plant network's gain they keep.

## Confirmatory and exploratory results

We planned the experiments one result at a time and chose later ones by their results on ISIC. From partway through the
study, we wrote down our predictions before each run. Five headline claims were fixed before we tested them on
BCN20000, a dermoscopy dataset we had not used until then. Those five are the study's confirmatory results, and
everything else is exploratory.

## The benchmark

PlantVillage is one of the BSCD-FSL benchmark's targets, so there we use a synthetic-pair start trained on 2,700
COIL-100 views alone, with six seeds, and compare it with plain ImageNet DINO ViT-S/16. The benchmark has four targets,
ISIC, ChestX, EuroSAT and CropDisease. Each one has 600 5-way episodes with 15 queries per class, and both networks see
the same episodes.

Each network adapts to each episode in one of three ways. A linear probe trains only a classifier. LoRA adds rank-16
adapters (alpha 16) to the query and value projections of all 12 blocks. Full fine-tuning trains either the last 6
blocks or all 12. LoRA and full fine-tuning also train the final norm and a head that starts from the linear probe.
Both use SGD (momentum 0.9, weight decay 1e-4, cosine schedule, gradient clipping at 1), on support images that are
cropped and flipped afresh at every step.

Every setting was chosen on plain alone, and only on development domains, so no setting was tuned on a test target.
PAD-UFES-20 stood in for ISIC, and CUB and Places for the other targets. Before running any target, we fixed full
fine-tuning at 5 shots as the primary comparison.

We also compare with REAP. Its supervised stage, from its released code, trains with 16 random registers on
miniImageNet's 64 training classes, once from plain and once from each seed of the synthetic-pair start. We keep the
last epoch rather than the checkpoint that does best on a target. REAP's frozen prototype test then runs on 4,000
episodes per target and shot.

## Where each part lives in the code

| Part | File |
|---|---|
| Sources, source training, few-shot evaluation and attribute distance | `code/e1_screen.py` |
| The confirmatory test on BCN20000 | `code/e2_confirmatory_bcn20000.py` |
| The label-free linear map | `code/e9_linear_stand_in.py` |
| Synthetic lesions and synthetic-pair training | `code/e15_pointing.py` |
| The benchmark | `code/e16_adaptation.py` |
| Networks and their readouts, datasets, few-shot classifiers, the bootstrap, the linear map and the benchmark's episodes | `code/common/` |

The paper's other checks, such as the attribute probes, cropping and the larger networks, are in our full codebase.
`DATA.md` lists the datasets and networks.
