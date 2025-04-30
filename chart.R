library(data.table)
library(jsonlite)
library(ggplot2)
library(scales)

setwd("/git/State-Transcription-Benchmark/results/")

filename_model_mapping = c(
  "assembly-ai-automatic-detection.json"="AssemblyAI",
  "assembly-ai.json"="AssemblyAI",
  "openai_gpt-4o-transcribe-automatic-detection.json"="GPT-4o",
  "openai_gpt-4o-transcribe.json"="GPT-4o",
  "openai_whisper-1-automatic-detection.json"="Whisper-1",
  "openai_whisper-1.json"="Whisper-1"
)
filename_method_mapping = c(
  "assembly-ai-automatic-detection.json"="Automatic",
  "assembly-ai.json"="Manual",
  "openai_gpt-4o-transcribe-automatic-detection.json"="Automatic",
  "openai_gpt-4o-transcribe.json"="Manual",
  "openai_whisper-1-automatic-detection.json"="Automatic",
  "openai_whisper-1.json"="Manual"
)

dat_list = list()
for(filename in names(filename_model_mapping)){
  json_data = fromJSON(filename)
  json_df = data.frame(locale="all", n=sum(json_data$locales$n), label="Maryland-weighted", wer=json_data$wer)
  json_df = rbind(json_df, json_data$locales)
  json_df$model = filename_model_mapping[filename]
  json_df$method = filename_method_mapping[filename]
  dat_list[[filename]] = json_df
}

dat = rbindlist(dat_list)

# Overall Maryland-weighted performance

overall = subset(dat, locale=="all")
overall = overall[order(overall$wer),]
overall$model = factor(overall$model, levels=unique(overall$model))
overall$method = factor(overall$method, levels=c("Manual","Automatic"))

overall$text_col = "white"
overall$text_col[which(overall$method=="Manual")] = "black"
 
cols = c(
  "#c94a46", "#a89474", "#050d5e"
)
  

ggplot(overall, aes(x=model, y=wer, group=method, fill=method)) +
  geom_bar(stat="identity", position="dodge", width=0.9) +
  geom_text(aes(label=percent_format(0.1)(wer), y=wer/2, color=text_col), position = position_dodge(width = .9), show.legend=F) +
  scale_y_continuous(labels=percent, expand=c(0, 0), n.breaks=6) +
  scale_fill_manual(values=c(cols[1],cols[3])) +
  scale_color_identity() +
  theme_bw() +
  theme(
    panel.border = element_blank()
    ,panel.grid.major.y = element_line(colour = "grey80")
    ,panel.grid.minor.y = element_blank()
    ,panel.grid.major.x = element_blank()
    ,panel.grid.minor.x = element_blank()
    ,panel.background = element_blank()
    ,plot.background = element_blank()
    ,axis.line.x = element_line(colour = "black")
    ,axis.line.y = element_blank()
    ,axis.ticks = element_blank()
    ,axis.text = element_text(size=13)
    ,axis.title = element_text(size=13)
    ,legend.text = element_text(size=13)
  ) +
  labs(
    title=paste0(
      "Transcription model performance on Maryland-weighted voice benchmark (n=",
      number_format(big.mark=",")(overall$n[1]),
      ")"
    ),
    fill="Language detection method", 
    x="", 
    y="Word error rate\n(lower is better)"
  )
ggsave("../graphics/overall.png",height=5,width=10)

# English performance

english = subset(dat, locale=="en")
english = english[order(english$wer),]
english$model = factor(english$model, levels=unique(english$model))
english$method = factor(english$method, levels=c("Manual","Automatic"))

english$text_col = "white"
english$text_col[which(english$method=="Manual")] = "black"


ggplot(english, aes(x=model, y=wer, group=method, fill=method)) +
  geom_bar(stat="identity", position="dodge", width=0.9) +
  geom_text(aes(label=percent_format(0.1)(wer), y=wer/2, color=text_col), position = position_dodge(width = .9), show.legend=F) +
  scale_y_continuous(labels=percent, expand=c(0, 0), n.breaks=6) +
  scale_fill_manual(values=c(cols[1],cols[3])) +
  scale_color_identity() +
  theme_bw() +
  theme(
    panel.border = element_blank()
    ,panel.grid.major.y = element_line(colour = "grey80")
    ,panel.grid.minor.y = element_blank()
    ,panel.grid.major.x = element_blank()
    ,panel.grid.minor.x = element_blank()
    ,panel.background = element_blank()
    ,plot.background = element_blank()
    ,axis.line.x = element_line(colour = "black")
    ,axis.line.y = element_blank()
    ,axis.ticks = element_blank()
    ,axis.text = element_text(size=13)
    ,axis.title = element_text(size=13)
    ,legend.text = element_text(size=13)
  ) +
  labs(
    title=paste0(
      "Transcription model performance on English voice benchmark (n=",
      number_format(big.mark=",")(english$n[1]),
      ")"
    ),
    fill="Language detection method", 
    x="", 
    y="Word error rate\n(lower is better)"
  )
ggsave("../graphics/english.png",height=5,width=10)

# Spanish performance

spanish = subset(dat, locale=="es")
spanish = spanish[order(spanish$wer),]
spanish$model = factor(spanish$model, levels=unique(spanish$model))
spanish$method = factor(spanish$method, levels=c("Manual","Automatic"))

spanish$text_col = "white"
spanish$text_col[which(spanish$method=="Manual")] = "black"

ggplot(spanish, aes(x=model, y=wer, group=method, fill=method)) +
  geom_bar(stat="identity", position="dodge", width=0.9) +
  geom_text(aes(label=percent_format(0.1)(wer), y=wer/2, color=text_col), position = position_dodge(width = .9), show.legend=F) +
  scale_y_continuous(labels=percent, expand=c(0, 0), n.breaks=6) +
  scale_fill_manual(values=c(cols[1],cols[3])) +
  scale_color_identity() +
  theme_bw() +
  theme(
    panel.border = element_blank()
    ,panel.grid.major.y = element_line(colour = "grey80")
    ,panel.grid.minor.y = element_blank()
    ,panel.grid.major.x = element_blank()
    ,panel.grid.minor.x = element_blank()
    ,panel.background = element_blank()
    ,plot.background = element_blank()
    ,axis.line.x = element_line(colour = "black")
    ,axis.line.y = element_blank()
    ,axis.ticks = element_blank()
    ,axis.text = element_text(size=13)
    ,axis.title = element_text(size=13)
    ,legend.text = element_text(size=13)
  ) +
  labs(
    title=paste0(
      "Transcription model performance on Spanish voice benchmark (n=",
      number_format(big.mark=",")(spanish$n[1]),
      ")"
    ),
    fill="Language detection method", 
    x="", 
    y="Word error rate\n(Lower is better)"
  )
ggsave("../graphics/spanish.png",height=5,width=10)

# All other languages performance

other = subset(dat, !locale %in% c("all", "en", "es") & method=="Manual")
other = other[order(other$wer),]
other$model = factor(other$model, levels=c("GPT-4o", "AssemblyAI", "Whisper-1"))
other$locale = factor(other$locale, levels=unique(other$locale))


ggplot(other, aes(x=locale, y=wer, group=model, fill=model)) +
  geom_bar(stat="identity", position="dodge", width=0.9) +
  scale_y_continuous(labels=percent, expand=c(0, 0), n.breaks=6) +
  scale_fill_manual(values=cols) +
  scale_color_identity() +
  theme_bw() +
  theme(
    panel.border = element_blank()
    ,panel.grid.major.y = element_line(colour = "grey80")
    ,panel.grid.minor.y = element_blank()
    ,panel.grid.major.x = element_blank()
    ,panel.grid.minor.x = element_blank()
    ,panel.background = element_blank()
    ,plot.background = element_blank()
    ,axis.line.x = element_line(colour = "black")
    ,axis.line.y = element_blank()
    ,axis.ticks.y = element_blank()
    ,axis.text = element_text(size=13)
    ,axis.title = element_text(size=13)
    ,legend.text = element_text(size=13)
    ,axis.text.x = element_text(angle = 90, vjust = 0.3, hjust=1, size=10)
  ) +
  labs(
    title=paste0(
      "Transcription model performance on manual-detection other locale voice benchmark (n=",
      number_format(big.mark=",")(sum(other$n) / 3),
      ")"
    ),
    fill="", 
    x="", 
    y="Word error rate\n(lower is better)"
  )
ggsave("../graphics/other.png",height=5,width=10)

