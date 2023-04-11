suppressPackageStartupMessages(library(ComplexHeatmap))
library(pals)
setwd("~/Projects/Github/PyComplexHeatmap/comparison")

# Read data
beta = read.table("beta.csv",sep="\t",row.names = 1,check.names = F)
row_info = read.table("df_row.csv",sep="\t",row.names = 1,check.names = F)
col_info = read.table("df_col.csv",sep="\t",row.names = 1,check.names = F)

# define color mapping
tissue_col <- structure(c("#00E5FF","#6CBF00","#007F19","#FF0000"),names=c('Frontal Lobe Brain','Liver','Tail','Spleen'))
strain_col <- structure(c('#66AA9F','#8A6699','#D8A49C'),names=c('CAST_EiJ','MOLF_EiJ','PWK_PhJ'))

target_col <- structure(c('yellowgreen','orangered'), names = c(0,1))
group_col <- structure(c('darkorange','skyblue','red','wheat','green','darkgray'),
                       names=c('Artificial high meth. reading','Artificial low meth. reading','G-R','No Effect','R-G','Suboptimal hybridization'))

# create heatmap annotations
col_ha <- HeatmapAnnotation(
    df=col_info[colnames(beta),c('Strain','Tissue')],
    col=list(Strain=strain_col,Tissue=tissue_col),
    show_legend = c(Strain=T,Tissue=T),
    annotation_name_side='right',simple_anno_size_adjust = TRUE,
    simple_anno_size = unit(3,'mm'))

row_ha <- HeatmapAnnotation(which='row',
                            df=row_info[rownames(beta),c('Target','Group')],
                            col=list('Target'=target_col,'Group'=group_col),
                            show_legend = T,simple_anno_size_adjust = TRUE,
                            annotation_name_side='top',
                            simple_anno_size = unit(3,'mm'))

col=colorRampPalette(c("blue","yellow"))(10)
# par(family="Arial")
pdf("ComplexHeatmap.pdf",family = "Helvetica")
hm2 <- Heatmap(
    as.matrix(beta),name='beta',
    cluster_columns = T,cluster_rows = T,
    cluster_row_slices = T,
    show_column_dend = F,show_row_dend=F,
    show_column_names = F,show_row_names = F,
    clustering_method_columns = "average",clustering_distance_columns = "pearson",
    clustering_method_rows = "average",clustering_distance_rows = "pearson",
    top_annotation = col_ha,
    left_annotation=row_ha,
    row_split = row_info[,c('Target','Group')],row_gap = unit(0,'mm'),
    column_split = col_info$Strain,column_gap=unit(0.2,'mm'),
    row_title = NULL,column_title = NULL,
    col=parula(n=9),
    na_col = "gray",
    width = unit(6, "cm"),height = unit(14,'cm'),
    use_raster = T,raster_quality = 2)

ht <- draw(hm2)
dev.off()